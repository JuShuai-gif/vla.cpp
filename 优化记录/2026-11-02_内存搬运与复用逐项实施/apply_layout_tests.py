from pathlib import Path
r=Path('/home/skysi/code/vla.cpp')
p=r/'tests/test_musa_ops.cpp';s=p.read_text()
s=s.replace('int64_t heads=4, int64_t nv=2) {','int64_t heads=4, int64_t nv=2, bool force_cast=false) {',1)
s=s.replace('''    auto * out = flash
        ? vla::flash_attention(g.ctx,view(q),view(k),view(v),nullptr,1.f/std::sqrt(float(hd)))''','''    auto * kview=view(k), * vview=view(v);
    if(force_cast) { kview=ggml_cast(g.ctx,kview,GGML_TYPE_F16);vview=ggml_cast(g.ctx,vview,GGML_TYPE_F16); }
    auto * out = flash
        ? vla::flash_attention(g.ctx,view(q),kview,vview,nullptr,1.f/std::sqrt(float(hd)))''',1)
key='''    if (argc == 2 && std::strcmp(argv[1], "--vision-sdpa") == 0) {'''
val='''    if (argc == 2 && std::strcmp(argv[1], "--vision-pack") == 0) {
        if(!vla::musa_vision_pack_enabled()) return 2;
        for(int64_t batch : {1,2})
            ok=compare("vision_pack_exact",GGML_TYPE_F16,
                vision_attention(b.handle,true,16,batch,true),
                vision_attention(b.handle,true,16,batch,false),1e-12) && ok;
        ggml_backend_free(cpu);ggml_backend_free(b.handle);
        return ok?0:1;
    }
'''+key
assert key in s;s=s.replace(key,val,1);p.write_text(s)
p=r/'tests/CMakeLists.txt';s=p.read_text()
key='    add_test(NAME musa_mapped_io COMMAND test_musa_ops --mapped-io)'
assert key in s
s=s.replace(key,key+'''
    add_test(NAME musa_vision_pack COMMAND test_musa_ops --vision-pack)
    set_tests_properties(musa_vision_pack PROPERTIES ENVIRONMENT "VLA_MUSA_VISION_SDPA=1;VLA_MUSA_VISION_PACK=1")
    add_test(NAME musa_attention256_direct COMMAND test_musa_ops --attention256)
    set_tests_properties(musa_attention256_direct PROPERTIES ENVIRONMENT "VLA_MUSA_GEMMA_ATTN=fused-all;VLA_MUSA_ATTN_DIRECT_OUT=1")
''',1);p.write_text(s)
