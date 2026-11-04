#include "gguf_reader.h"
#include "loader.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <vector>

int main() {
    const auto path=std::filesystem::temp_directory_path()/
        ("vla-load-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".gguf");
    const int64_t n=1048579; // SIMD-aligned chunks plus a scalar tail.
    auto * ctx=ggml_init({(size_t)n*8+1024*1024,nullptr,false});
    auto * gguf=gguf_init_empty();
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
        auto * t=ggml_new_tensor_1d(ctx,type,n);ggml_set_name(t,ggml_type_name(type));
        if(type==GGML_TYPE_F32) {
            auto * dst=(uint32_t *)t->data;
            for(int64_t i=0;i<n;++i)dst[i]=(uint32_t)i*1664525u+1013904223u;
        } else {
            auto * dst=(uint16_t *)t->data;
            for(int64_t i=0;i<n;++i)dst[i]=(uint16_t)i; // Includes all NaN/subnormal patterns.
        }
        gguf_add_tensor(gguf,t);
    }
    if(!gguf_write_to_file(gguf,path.string().c_str(),false))return 1;
    gguf_free(gguf);ggml_free(ctx);
    bool ok=true;
    {
        vla::gguf_reader reader("load-test");
        if(!reader.open(path.string()))return 1;
        for(auto source:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16})
            for(auto target:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16})
                for(bool norm:{false,true}) {
                    auto f=reader.read_f32(ggml_type_name(source));
                    if(norm)for(auto & x:f)x+=1.f;
                    std::vector<uint8_t> ref((size_t)n*ggml_type_size(target));
                    if(target==GGML_TYPE_F32)std::memcpy(ref.data(),f.data(),ref.size());
                    else if(target==GGML_TYPE_F16)ggml_fp32_to_fp16_row(f.data(),(ggml_fp16_t *)ref.data(),n);
                    else ggml_fp32_to_bf16_row(f.data(),(ggml_bf16_t *)ref.data(),n);
                    for(const char * threads:{"1","4"}) {
#ifdef _WIN32
                        _putenv_s("VLA_LOAD_THREADS",threads);
#else
                        setenv("VLA_LOAD_THREADS",threads,1);
#endif
                        auto got=reader.read_convert(ggml_type_name(source),target,norm);
                        if(got!=ref){std::fprintf(stderr,"mismatch %s -> %s norm=%d threads=%s\\n",
                            ggml_type_name(source),ggml_type_name(target),norm,threads);ok=false;}
                    }
                }
        auto * backend=ggml_backend_cpu_init();
        for(const char * direct:{"0","1"}) {
#ifdef _WIN32
            _putenv_s("VLA_LOAD_DIRECT_FUSED",direct);
#else
            setenv("VLA_LOAD_DIRECT_FUSED",direct,1);
#endif
            auto * weights=ggml_init({64*ggml_tensor_overhead(),nullptr,true});
            vla::WeightLoader loader("load-test",reader,weights,GGML_TYPE_F16);
            auto * merged=loader.fuse_typed(GGML_TYPE_F16,"merged",{"bf16","f32"});
            ggml_backend_buffer_t buffer=nullptr;
            if(!merged || !loader.upload(backend,&buffer))return 1;
            auto ref=reader.read_convert("bf16",GGML_TYPE_F16);
            auto tail=reader.read_convert("f32",GGML_TYPE_F16);ref.insert(ref.end(),tail.begin(),tail.end());
            std::vector<uint8_t> got(ggml_nbytes(merged));ggml_backend_tensor_get(merged,got.data(),0,got.size());
            if(got!=ref)ok=false;
            ggml_backend_buffer_free(buffer);ggml_free(weights);
        }
        ggml_backend_free(backend);
    }
    std::filesystem::remove(path);
    return ok?0:1;
}
