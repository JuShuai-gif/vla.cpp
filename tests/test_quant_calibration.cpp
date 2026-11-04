// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include "quant_calibration.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#define CHECK(expr) do { if (!(expr)) {std::fprintf(stderr,"CHECK failed at line %d: %s\n",__LINE__,#expr);return 1;} } while(0)

int main(int argc,char **argv) {
    if(argc!=2)return 2;
    const std::filesystem::path root=argv[1];
    if(!std::filesystem::create_directory(root))return 2;
    setenv("VLA_PI05_CALIBRATION_DIR",root.c_str(),1);
    auto *ctx=ggml_init({1024*1024,nullptr,true});
    auto *w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,64);
    auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2);
    ggml_set_name(w,"vlm.blk.0.ffn_gate.weight");
    auto *y=ggml_mul_mat(ctx,w,x);
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    vla::mark_quant_calibration(graph);
    CHECK(x->flags&GGML_TENSOR_FLAG_OUTPUT);
    auto *cpu=ggml_backend_cpu_init();
    auto *buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);
    float values[64];for(int j=0;j<64;++j)values[j]=float(j%32+j/32);
    ggml_backend_tensor_set(x,values,0,sizeof(values));
    CHECK(vla::write_quant_calibration("prefix",graph,0,ggml_graph_n_nodes(graph)));
    const auto captured=root/"0-prefix.jsonl";
    {std::ifstream in(captured);std::string text((std::istreambuf_iterator<char>(in)),{});
     std::ostringstream maximum,squares;maximum<<"\"amax\":[";squares<<"\"sum_squares\":[";
     for(int j=0;j<32;++j){maximum<<(j?",":"")<<j+1;squares<<(j?",":"")<<j*j+(j+1)*(j+1);}maximum<<"]";squares<<"]";
     CHECK(text.find(maximum.str())!=std::string::npos && text.find(squares.str())!=std::string::npos);
     CHECK(text.find("\"rows\":2")!=std::string::npos);}
    const auto retained=root/"1-prefix.jsonl";
    {std::ofstream out(retained);out<<"preserve";}
    CHECK(!vla::write_quant_calibration("prefix",graph,0,ggml_graph_n_nodes(graph)));
    {std::ifstream in(retained);std::string contents;in>>contents;CHECK(contents=="preserve");}
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(cpu);
    unsetenv("VLA_PI05_CALIBRATION_DIR");
    std::filesystem::remove(captured);std::filesystem::remove(retained);std::filesystem::remove(root);
    return 0;
}
