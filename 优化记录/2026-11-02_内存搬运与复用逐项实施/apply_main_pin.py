from pathlib import Path
r=Path('/home/skysi/code/vla.cpp');p=r/'src/models/pi05.cpp';s=p.read_text()
s=s.replace('bool async_vision = false, direct_vision = false;','bool async_vision = false, direct_vision = false, pin_main_input = false;',1)
s=s.replace('ggml_backend_buffer_t pixels_host = nullptr;','ggml_backend_buffer_t main_host = nullptr;\n    size_t main_host_bytes = 0;\n    ggml_backend_buffer_t pixels_host = nullptr;',1)
s=s.replace('    if (pixels_host) ggml_backend_buffer_free(pixels_host);','    if (main_host) ggml_backend_buffer_free(main_host);\n    if (pixels_host) ggml_backend_buffer_free(pixels_host);',1)
s=s.replace('        m->direct_vision = b.is_musa && env_flag("VLA_PI05_VISION_DIRECT");','        m->direct_vision = b.is_musa && env_flag("VLA_PI05_VISION_DIRECT");\n        m->pin_main_input = b.is_musa && env_flag("VLA_PI05_PIN_MAIN_INPUT");',1)
s=s.replace('m->input_cache || m->async_vision || m->direct_vision ||','m->input_cache || m->async_vision || m->direct_vision || m->pin_main_input ||',1)
anchor='''    if (in.precomputed_img_emb) {
        ggml_backend_tensor_set(t_image_emb,'''
addition='''    float * main_staging = nullptr;
#ifdef GGML_USE_MUSA
    if (pin_main_input) {
        // Allocate before vision submission; steady-state uploads/readback share one persistent slab.
        const size_t bytes=(lang_rows.size()+2*(size_t)max_ad*chunk)*sizeof(float);
        if (!main_host || main_host_bytes<bytes) {
            if(main_host) ggml_backend_buffer_free(main_host);
            main_host=ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(),bytes);
            main_host_bytes=main_host?bytes:0;
        }
        if(!main_host) return {};
        main_staging=(float *)ggml_backend_buffer_get_base(main_host);
    }
#endif
'''+anchor
assert anchor in s;s=s.replace(anchor,addition,1)
anchor='''    if(input_cache) ggml_backend_tensor_set_async(backend,t_prefix,lang_rows.data(),
        (size_t)n_img_tokens*t_prefix->nb[1],lang_rows.size()*sizeof(float));'''
replace='''    if(main_staging) std::memcpy(main_staging,lang_rows.data(),lang_rows.size()*sizeof(float));
    if(input_cache) ggml_backend_tensor_set_async(backend,t_prefix,main_staging?main_staging:lang_rows.data(),
        (size_t)n_img_tokens*t_prefix->nb[1],lang_rows.size()*sizeof(float));'''
assert anchor in s;s=s.replace(anchor,replace,1)
anchor='''    if(input_cache) ggml_backend_tensor_set_async(backend,t_x0,x0h.data(),0,ggml_nbytes(t_x0));'''
replace='''    float * noise_staging=main_staging?main_staging+lang_rows.size():nullptr;
    if(noise_staging) std::memcpy(noise_staging,x0h.data(),ggml_nbytes(t_x0));
    if(input_cache) ggml_backend_tensor_set_async(backend,t_x0,noise_staging?noise_staging:x0h.data(),0,ggml_nbytes(t_x0));'''
assert anchor in s;s=s.replace(anchor,replace,1)
s=s.replace('ggml_backend_tensor_get_async(backend,x_final,out.data(),0,out.size()*sizeof(float));',
'''ggml_backend_tensor_get_async(backend,x_final,noise_staging?noise_staging+x0h.size():out.data(),0,out.size()*sizeof(float));''',1)
s=s.replace('''    if (!output_ready) ggml_backend_tensor_get(x_final, out.data(), 0, out.size()*sizeof(float));''','''    if (output_ready && noise_staging) std::memcpy(out.data(),noise_staging+x0h.size(),out.size()*sizeof(float));
    if (!output_ready) ggml_backend_tensor_get(x_final, out.data(), 0, out.size()*sizeof(float));''',1)
p.write_text(s)
p=r/'scripts/bench_pi05.py';s=p.read_text().replace("'VLA_PI05_HOST_INPUT',","'VLA_PI05_PIN_MAIN_INPUT', 'VLA_PI05_HOST_INPUT',",1);p.write_text(s)
print('main staging candidate applied')
