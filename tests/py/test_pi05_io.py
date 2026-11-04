import importlib.util
from pathlib import Path
import sys,tempfile,unittest
try:
    import numpy as np
    import gguf
    from PIL import Image
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise unittest.SkipTest("pi05 IO tests need converter/image dependencies") from exc
# Other converter tests intentionally install lightweight module stubs on bare CI.
if not (hasattr(np,"random") and hasattr(gguf,"ReaderField") and hasattr(torch,"set_num_threads")):
    raise unittest.SkipTest("pi05 IO tests need real converter/image dependencies")
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'scripts'))
from pi05_gguf import GGUFReader
from pi05_preprocess import prepare_image,prepare_images,prepare_rgb
torch.set_num_threads(4)

def reference(path,size):
    with Image.open(path)as im:rgb=np.asarray(im.convert("RGB"),dtype=np.float32)/255.0
    h,w=rgb.shape[:2]
    if(h,w)!=(size,size):
        ratio=max(w/size,h/size);rh,rw=max(1,int(h/ratio)),max(1,int(w/ratio))
        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1)[None]
        tensor=F.interpolate(tensor,size=(rh,rw),mode="bilinear",align_corners=False).clamp(0,1)
        top,left=(size-rh)//2,(size-rw)//2
        tensor=F.pad(tensor,(left,size-rw-left,top,size-rh-top))
        rgb=tensor[0].permute(1,2,0).numpy()
    return np.ascontiguousarray(rgb,dtype=np.float32)

class Images(unittest.TestCase):
    def test_pixels_and_buffers(self):
        rng=np.random.default_rng(41)
        with tempfile.TemporaryDirectory()as temp:
            for h,w in [(480,640),(640,480),(224,224),(225,223),(1,17),(17,1),(7,11),(1025,127)]:
                a=rng.integers(0,256,(h,w,3),dtype=np.uint8)
                if h>1:a[0]=0;a[-1]=255
                path=Path(temp)/'rgb.png';Image.fromarray(a).save(path)
                expected=reference(path,224)
                np.testing.assert_array_equal(prepare_image(path,224),expected)
                out=np.full((224,224,3),np.nan,np.float32)
                self.assertIs(prepare_rgb(a,224,out),out)
                np.testing.assert_array_equal(out,expected)
                np.testing.assert_array_equal(prepare_images([path,path],224)[0],expected)
                # Camera views may have negative strides.
                path2=Path(temp)/'flip.png';Image.fromarray(a[:,::-1]).save(path2)
                np.testing.assert_array_equal(prepare_rgb(a[:,::-1],224,out),reference(path2,224))

    def test_camera_order_and_formats(self):
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory()as temp:
            paths=[]
            for mode,shape in [('L',(37,59)),('RGBA',(59,37,4)),('RGB',(480,640,3))]:
                a=rng.integers(0,256,shape,dtype=np.uint8);path=Path(temp)/(mode+'.png')
                Image.fromarray(a,mode=mode).save(path);paths.append(path)
            for path,a in zip(paths,prepare_images(paths,224)):
                np.testing.assert_array_equal(a,reference(path,224))
            jpeg=Path(temp)/'rgb.jpg';Image.open(paths[-1]).convert('RGB').save(jpeg)
            np.testing.assert_array_equal(prepare_image(jpeg,224),reference(jpeg,224))

    def test_validation(self):
        for a in [np.zeros((3,3),np.uint8),np.zeros((2,3,4),np.uint8),np.zeros((0,3,3),np.uint8),np.zeros((3,3,3),np.float32)]:
            with self.assertRaises(ValueError):prepare_rgb(a,224)
        with self.assertRaises(ValueError):prepare_rgb(np.zeros((2,2,3),np.uint8),224,[])

class Metadata(unittest.TestCase):
    def test_contents_endian_and_payload(self):
        for endian in [gguf.GGUFEndian.LITTLE,gguf.GGUFEndian.BIG]:
            with tempfile.TemporaryDirectory()as temp:
                path=Path(temp)/'test.gguf'
                w=gguf.GGUFWriter(path,'pi05',endianess=endian)
                w.add_key_value('bytes',np.arange(65537,dtype=np.uint8).tolist(),gguf.GGUFValueType.ARRAY,sub_type=gguf.GGUFValueType.UINT8)
                w.add_array('numbers',[1,-4,17])
                w.add_array('byte_values',[0,128,255])
                w.add_array('floats',[.1,.5,-1.25])
                w.add_array('strings',['top','wrist'])
                w.add_tensor('state_q01',np.arange(7,dtype=np.float32));w.write_header_to_file()
                w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                old=gguf.GGUFReader(path);fast=GGUFReader(path)
                for name in old.fields:
                    self.assertEqual(old.fields[name].contents(),fast.fields[name].contents())
                for key in ['bytes','numbers','floats','strings']:
                    a,b=old.fields[key],fast.fields[key]
                    for ix in [0,-1,False,True,slice(1,3),slice(None,None,-1)]:
                        self.assertEqual(a.contents(ix),b.contents(ix))
                self.assertEqual(fast.fields['bytes'].raw_bytes(),bytes(old.fields['bytes'].contents()))
                self.assertEqual(fast.fields['byte_values'].raw_bytes(),bytes(old.fields['byte_values'].contents()))
                with self.assertRaises(ValueError):fast.fields['numbers'].raw_bytes()
                self.assertEqual(old.tensors[0].data.tobytes(),fast.tensors[0].data.tobytes())
                # Reject huge/truncated numeric arrays without per-element allocation.
                f=fast.fields['bytes'];raw=bytearray(path.read_bytes())
                length_offset=f.offset+8+len('bytes')+4+4
                raw[length_offset:length_offset+8]=(2**62).to_bytes(8,'big'if endian==gguf.GGUFEndian.BIG else 'little')
                bad=Path(temp)/'bad.gguf';bad.write_bytes(raw)
                with self.assertRaises((ValueError,OverflowError)):GGUFReader(bad)

if __name__=='__main__':unittest.main()
