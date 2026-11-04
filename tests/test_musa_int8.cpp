// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include "backend.h"
#include "layers/musa_int8.h"
#include "musa/vla_musa_ops.h"
#include "ggml-alloc.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <limits>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(){
    const auto b=vla::backend_init("native-int8-test",4);
    if(!b.handle || !b.is_musa)return 77;
    CHECK(vla::musa_register_fused_ops() && vla::musa_small_gemm_supported());
    const int K=96,M=128;
    const auto root=std::filesystem::temp_directory_path()/("vla-int8-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(root));
    auto *source=ggml_init({1024*1024,nullptr,false});
    auto *w=ggml_new_tensor_2d(source,GGML_TYPE_F16,K,M);ggml_set_name(w,"weight");
    auto *wp=(ggml_fp16_t*)w->data;std::vector<float> weights(K*M);
    for(int i=0;i<K*M;++i){wp[i]=ggml_fp32_to_fp16(float((i*17)%101-50)/71);weights[i]=ggml_fp16_to_fp32(wp[i]);}
    auto *file=gguf_init_empty();gguf_add_tensor(file,w);CHECK(gguf_write_to_file(file,(root/"model.gguf").c_str(),false));
    gguf_free(file);ggml_free(source);
    vla::gguf_reader reader("int8-test");CHECK(reader.open((root/"model.gguf").string()));
    auto rejects=[&](const std::vector<float>& inv,const std::vector<int32_t>& indices,bool trailing=false){
        {std::ofstream f(root/"params",std::ios::binary);uint32_t h[]={0x31503849,K,(uint32_t)indices.size()};
         f.write((char*)h,sizeof(h));f.write((char*)inv.data(),inv.size()*4);f.write((char*)indices.data(),indices.size()*4);
         if(trailing)f.put('x');}
        auto *ctx=ggml_init({1024*1024,nullptr,true});
        vla::WeightLoader loader("invalid-parameter-test",reader,ctx,GGML_TYPE_F16);
        auto bad=loader.quant_i8_compensated("bad",{"weight"},(root/"params").string());
        const bool rejected=!bad;ggml_free(ctx);return rejected;
    };
    std::vector<float> invalid(K,1);
    for(float value:{0.f,-1.f,1e-20f,1e20f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        invalid[0]=value;CHECK(rejects(invalid,{}));
    }
    invalid[0]=0;CHECK(rejects(invalid,{0,0}));CHECK(rejects(invalid,{K}));CHECK(rejects(invalid,{-1}));
    invalid[0]=1;CHECK(rejects(invalid,{},true));invalid.pop_back();CHECK(rejects(invalid,{}));
    for(int R:{0,16}){
        std::vector<float> inv(K);for(int k=0;k<K;++k)inv[k]=k<R?0:float(k%5+1)/3;
        std::vector<int32_t> indices(R);for(int i=0;i<R;++i)indices[i]=i;
        {std::ofstream f(root/"params",std::ios::binary);uint32_t h[]={0x31503849,K,(uint32_t)R};
         f.write((char*)h,sizeof(h));f.write((char*)inv.data(),K*4);f.write((char*)indices.data(),R*4);}
        auto *wc=ggml_init({1024*1024,nullptr,true});
        vla::WeightLoader loader("test",reader,wc,GGML_TYPE_F16);
        auto q=loader.quant_i8_compensated("test.int8",{"weight"},(root/"params").string());
        ggml_backend_buffer_t wb=nullptr;CHECK(q && loader.upload(b.handle,&wb));
        std::vector<int8_t> codes(M*K),packed(M*K),actual(M*K);std::vector<float> sw(M);
        for(int m=0;m<M;++m){
            float mx=0;for(int k=0;k<K;++k)if(inv[k])mx=std::max(mx,std::fabs(weights[m*K+k]/inv[k]));
            sw[m]=mx?mx/127:1;
            for(int k=0;k<K;++k){
                codes[m*K+k]=inv[k]?(int8_t)std::lrint(std::min(127.f,std::max(-127.f,weights[m*K+k]/inv[k]/sw[m]))):0;
                size_t dest=((m/32)*(K/32)+k/32)*1024+(m%32/16)*512+((m%16)*4+(k%16)/4+(k%32/16)*64)*4+k%4;
                packed[dest]=codes[m*K+k];
            }
        }
        ggml_backend_tensor_get(q.w,actual.data(),0,actual.size());CHECK(actual==packed);
        for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16})for(int N:{1,50,64,65,161}){
            auto *ctx=ggml_init({1024*1024,nullptr,true});auto *x=ggml_new_tensor_2d(ctx,type,K,N);ggml_set_input(x);
            auto *out=vla::musa_int8_linear(ctx,q,x);ggml_set_output(out);
            auto *fused=R==0 && N<=64?vla::musa_int8_linear(ctx,q,x,true):nullptr;
            if(fused)ggml_set_output(fused->src[0]);
            auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
            if(fused)ggml_build_forward_expand(graph,fused);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)CHECK(ggml_backend_supports_op(b.handle,ggml_graph_node(graph,i)));
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(b.handle));CHECK(ggml_gallocr_alloc_graph(allocator,graph));
            std::vector<float> input(K*N),got(M*N);std::vector<ggml_fp16_t> half_input(K*N);double worst=0;
            for(int repeat=0;repeat<5;++repeat){
                for(int i=0;i<K*N;++i){
                    float value=i<K&&repeat==0?0:float((i*13+repeat*7)%67-33)/11;
                    if(i%K==19)value*=13;
                    half_input[i]=ggml_fp32_to_fp16(value);input[i]=type==GGML_TYPE_F16?ggml_fp16_to_fp32(half_input[i]):value;
                }
                ggml_backend_tensor_set(x,type==GGML_TYPE_F16?(void*)half_input.data():(void*)input.data(),0,ggml_nbytes(x));
                CHECK(ggml_backend_graph_compute(b.handle,graph)==GGML_STATUS_SUCCESS);
                ggml_backend_tensor_get(out,got.data(),0,got.size()*4);
                if(fused){
                    std::vector<ggml_fp16_t> values(M/2*64);
                    ggml_backend_tensor_get(fused->src[0],values.data(),0,values.size()*2);
                    for(int n=0;n<64;++n)for(int m=0;m<M/2;++m){
                        float ref=0;
                        if(n<N){
                            const float g=got[n*M+m],u=got[n*M+M/2+m];
                            ref=0.5f*g*(1.0f+std::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)))*u;
                            ref=ggml_fp16_to_fp32(ggml_fp32_to_fp16(ref));
                        }
                        const float value=ggml_fp16_to_fp32(values[n*M/2+m]);
                        CHECK(std::isfinite(value) && std::fabs(ref-value)<0.005f+0.002f*std::fabs(ref));
                        if(n>=N)CHECK(values[n*M/2+m]==0);
                    }
                }
                for(int n=0;n<N;++n){
                    float mx=0;for(int k=0;k<K;++k)mx=std::max(mx,std::fabs(input[n*K+k]*inv[k]));
                    const float sx=mx?mx/127:1;std::vector<int8_t> xq(K);
                    for(int k=0;k<K;++k)xq[k]=(int8_t)std::lrint(std::min(127.f,std::max(-127.f,input[n*K+k]*inv[k]/sx)));
                    for(int m=0;m<M;++m){
                        int32_t acc=0;for(int k=0;k<K;++k)acc+=(int32_t)codes[m*K+k]*xq[k];
                        float ref=float(acc)*sw[m]*sx;
                        for(int k=0;k<R;++k)ref+=weights[m*K+k]*ggml_fp16_to_fp32(ggml_fp32_to_fp16(input[n*K+k]));
                        const double error=std::fabs(ref-got[n*M+m]);worst=std::max(worst,error);
                        CHECK(std::isfinite(got[n*M+m]) && error<0.0005+0.00005*std::fabs(ref));
                    }
                }
            }
            printf("R=%d N=%d type=%s max_error=%.9g\n",R,N,ggml_type_name(type),worst);
            ggml_gallocr_free(allocator);ggml_free(ctx);
        }
        for(int N:{1,50,64}){
            auto *ctx=ggml_init({1024*1024,nullptr,true});
            auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N),*o=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);
            auto *gate=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3*K),*mod=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3*K);
            for(auto *input:{x,o,gate,mod})ggml_set_input(input);
            ggml_tensor *res0=nullptr,*res1=nullptr;
            auto *n0=vla::packed_norm(ctx,x,o,gate,mod,1e-6f,1,&res0);
            auto *n1=vla::packed_norm(ctx,x,o,gate,mod,1e-6f,1,&res1,q.inverse);
            auto *y0=vla::musa_int8_linear(ctx,q,n0),*y1=vla::musa_int8_linear(ctx,q,n1);
            CHECK(y1->src[2]->op==GGML_OP_VIEW);
            for(auto *out:{y0,y1,res0,res1})ggml_set_output(out);
            auto *graph=ggml_new_graph(ctx);
            for(auto *out:{y0,y1,res0,res1})ggml_build_forward_expand(graph,out);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(b.handle));CHECK(ggml_gallocr_alloc_graph(allocator,graph));
            for(int rep=0;rep<3;++rep){
                for(auto *input:{x,o,gate,mod}){
                    std::vector<float> values(ggml_nelements(input));
                    for(size_t i=0;i<values.size();++i)values[i]=float(int((i*7+rep*3)%31)-15)/91.f;
                    ggml_backend_tensor_set(input,values.data(),0,values.size()*4);
                }
                CHECK(ggml_backend_graph_compute(b.handle,graph)==GGML_STATUS_SUCCESS);
                for(auto pair:{std::pair<ggml_tensor*,ggml_tensor*>(y0,y1),{res0,res1}}){
                    std::vector<float> a(ggml_nelements(pair.first)),b(a.size());
                    ggml_backend_tensor_get(pair.first,a.data(),0,a.size()*4);
                    ggml_backend_tensor_get(pair.second,b.data(),0,b.size()*4);
                    CHECK(a==b);for(float v:b)CHECK(std::isfinite(v));
                }
            }
            printf("packed norm INT8 exact R=%d N=%d\n",R,N);
            ggml_gallocr_free(allocator);ggml_free(ctx);
        }
        ggml_backend_buffer_free(wb);ggml_free(wc);
    }
    ggml_backend_free(b.handle);
    std::filesystem::remove(root/"params");std::filesystem::remove(root/"model.gguf");std::filesystem::remove(root);
    return 0;
}
