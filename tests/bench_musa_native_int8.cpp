// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include "backend.h"
#include "layers/musa_int8.h"
#include "layers/small_gemm.h"
#include "layers/packed_norm.h"
#include "ggml-alloc.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(){
    const auto b=vla::backend_init("int8-production-gemm-bench",4);
    if(!b.handle || !b.is_musa)return 77;
    CHECK(vla::musa_register_fused_ops() && vla::musa_small_gemm_supported());
    const int K=1024,M=8192,N=50;
    const auto root=std::filesystem::temp_directory_path()/("vla-int8-bench-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(root));
    auto *source=ggml_init({size_t(K)*M*2+1024*1024,nullptr,false});
    auto *w=ggml_new_tensor_2d(source,GGML_TYPE_F16,K,M);ggml_set_name(w,"weight");
    for(int i=0;i<K*M;++i)((ggml_fp16_t*)w->data)[i]=ggml_fp32_to_fp16(float((i*17)%101-50)/71);
    auto *file=gguf_init_empty();gguf_add_tensor(file,w);CHECK(gguf_write_to_file(file,(root/"model.gguf").c_str(),false));
    gguf_free(file);ggml_free(source);
    {std::ofstream f(root/"params",std::ios::binary);uint32_t h[]={0x31503849,K,0};std::vector<float> inv(K,1);
     f.write((char*)h,sizeof(h));f.write((char*)inv.data(),K*4);}
    vla::gguf_reader reader("int8-bench");CHECK(reader.open((root/"model.gguf").string()));
    auto *wc=ggml_init({1024*1024,nullptr,true});
    vla::WeightLoader loader("int8-bench",reader,wc,GGML_TYPE_F16);
    auto q=loader.quant_i8_compensated("int8",{"weight"},(root/"params").string());
    auto *wf=loader.gemm("weight");ggml_backend_buffer_t wb=nullptr;CHECK(loader.upload(b.handle,&wb));
    std::vector<ggml_fp16_t> data(K*64);
    for(int i=0;i<K*64;++i)data[i]=ggml_fp32_to_fp16(float((i*13)%67-33)/11);
    for(int round=0;round<4;++round)for(int index=0;index<2;++index){
        int mode=round%2?1-index:index;
        auto *ctx=ggml_init({1024*1024,nullptr,true});auto *pad=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,K,64);ggml_set_input(pad);
        auto *x=ggml_view_2d(ctx,pad,K,N,pad->nb[1],0);
        auto *out=mode?vla::musa_int8_linear(ctx,q,x,true):vla::packed_geglu(ctx,vla::small_gemm(ctx,wf,x,32));
        ggml_set_output(out);auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(b.handle));CHECK(ggml_gallocr_alloc_graph(allocator,graph));
        std::vector<double> times;
        for(int rep=0;rep<70;++rep){
            ggml_backend_tensor_set(pad,data.data(),0,data.size()*2);
            auto begin=std::chrono::steady_clock::now();
            CHECK(ggml_backend_graph_compute(b.handle,graph)==GGML_STATUS_SUCCESS);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            if(rep>=20)times.push_back(ms);
        }
        printf(R"json({"round":%d,"mode":"%s","samples_ms":[)json",round,mode?"native_int8_geglu":"production_f16_small_gemm_geglu");
        for(size_t i=0;i<times.size();++i)printf("%s%.6f",i?",":"",times[i]);
        std::sort(times.begin(),times.end());printf(R"json(],"p50_ms":%.6f})json" "\n",times[times.size()/2]);
        ggml_gallocr_free(allocator);ggml_free(ctx);
    }
    ggml_backend_buffer_free(wb);ggml_free(wc);ggml_backend_free(b.handle);
    std::filesystem::remove(root/"params");std::filesystem::remove(root/"model.gguf");std::filesystem::remove(root);return 0;
}
