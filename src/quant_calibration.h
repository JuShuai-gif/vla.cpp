// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace vla {
// Diagnostic runs retain GEMM inputs; their latency and memory are not benchmarks.
inline const char * quant_calibration_dir() {
    const char * dir = std::getenv("VLA_PI05_CALIBRATION_DIR");
    return dir && *dir ? dir : nullptr;
}

inline bool calibration_linear(const ggml_tensor * t) {
    if (!t || (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_CUSTOM)) return false;
    const auto * w=t->src[0]; const auto * x=t->src[1];
    if (!w || !x || w->ne[2]!=1 || w->ne[3]!=1) return false;
    const std::string name=w->name;
    return name.find("vlm.blk.")==0 || name.find("aex.blk.")==0 || name.find("vit.blk.")==0;
}

inline bool calibration_supported(const ggml_tensor * x, const ggml_tensor * w) {
    return x && ggml_is_contiguous(x) && x->ne[0]==w->ne[0] &&
           (x->type==GGML_TYPE_F32 || x->type==GGML_TYPE_F16 || x->type==GGML_TYPE_BF16);
}

inline void mark_quant_calibration(ggml_cgraph * graph) {
    if (!quant_calibration_dir()) return;
    for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto * t=ggml_graph_node(graph,i);
        if (calibration_linear(t) && calibration_supported(t->src[1],t->src[0]))
            ggml_set_output(t->src[1]);
    }
}

inline std::string calibration_json_string(const char * value) {
    std::string out="\"";
    for (const unsigned char * p=(const unsigned char *)value;*p;++p) {
        if (*p=='"' || *p=='\\') out+='\\';
        if (*p<32) { char escape[7];std::snprintf(escape,sizeof(escape),"\\u%04x",*p);out+=escape; }
        else out+=char(*p);
    }
    return out+'"';
}

inline bool write_quant_calibration(const char * phase, ggml_cgraph * graph, int begin, int end) {
    const char * dir=quant_calibration_dir(); if (!dir) return true;
    // Public API permits one resident model; each fresh process uses its own directory.
    static unsigned long long serial=0;
    const std::string path=std::string(dir)+"/"+std::to_string(serial++)+"-"+phase+".jsonl";
    FILE * f=std::fopen(path.c_str(),"wx");
    if (!f) {std::fprintf(stderr,"vla: cannot exclusively create calibration file %s\n",path.c_str());return false;}
    std::map<std::string,int> occurrences;
    bool valid=true;
    for (int i=begin;i<end;++i) {
        auto * t=ggml_graph_node(graph,i); if (!calibration_linear(t)) continue;
        const auto * w=t->src[0]; const auto * x=t->src[1];
        const int occurrence=occurrences[w->name]++;
        std::fprintf(f,"{\"phase\":%s,\"node\":%d,\"weight\":%s,\"occurrence\":%d,\"input_type\":%s,",
            calibration_json_string(phase).c_str(),i,calibration_json_string(w->name).c_str(),occurrence,
            calibration_json_string(ggml_type_name(x->type)).c_str());
        if (!calibration_supported(x,w)) {
            std::fprintf(f,"\"status\":\"unsupported_packed_or_strided_input\"}\n");continue;
        }
        const size_t K=x->ne[0],available_rows=ggml_nelements(x)/K;
        const size_t rows=std::min(available_rows,size_t(ggml_nelements(t)/w->ne[1]));
        std::vector<float> values(ggml_nelements(x));
        if (x->type==GGML_TYPE_F32) ggml_backend_tensor_get(x,values.data(),0,ggml_nbytes(x));
        else {
            std::vector<uint16_t> packed(values.size());ggml_backend_tensor_get(x,packed.data(),0,ggml_nbytes(x));
            for (size_t j=0;j<values.size();++j) {
                if (x->type==GGML_TYPE_F16) values[j]=ggml_fp16_to_fp32(packed[j]);
                else { ggml_bf16_t value;value.bits=packed[j];values[j]=ggml_bf16_to_fp32(value); }
            }
        }
        // Optional diagnostic input replay; never used by ordinary inference.
        const char * capture=std::getenv("VLA_PI05_CALIBRATION_INPUTS");
        if (capture && (std::string(",")+capture+",").find(","+std::string(w->name)+",")!=std::string::npos) {
            const std::string input_path=path+"."+std::to_string(i)+".f32";
            FILE * input=std::fopen(input_path.c_str(),"wbx");
            if (!input) {valid=false;break;}
            const uint64_t shape[2]={rows,K};
            bool written=std::fwrite(shape,sizeof(shape),1,input)==1 &&
                         std::fwrite(values.data(),sizeof(float),rows*K,input)==rows*K;
            if (std::fclose(input)!=0) written=false;
            if (!written) {valid=false;break;}
            std::fprintf(f,"\"input_file\":%s,",calibration_json_string(input_path.c_str()).c_str());
        }
        std::vector<double> sum2(K,0.0),maximum(K,0.0);
        for (size_t j=0;j<rows*K;++j) {
            const double v=values[j]; if (!std::isfinite(v)) {valid=false;break;}
            sum2[j%K]+=v*v;maximum[j%K]=std::max(maximum[j%K],std::fabs(v));
        }
        if (!valid) {std::fprintf(f,"\"status\":\"nonfinite\"}\n");break;}
        std::fprintf(f,"\"status\":\"captured\",\"channels\":%zu,\"rows\":%zu,\"amax\":[",K,rows);
        for (size_t j=0;j<K;++j) std::fprintf(f,"%s%.9g",j?",":"",maximum[j]);
        std::fprintf(f,"],\"sum_squares\":[");
        for (size_t j=0;j<K;++j) std::fprintf(f,"%s%.17g",j?",":"",sum2[j]);
        std::fprintf(f,"]}\n");
    }
    if (std::ferror(f)) valid=false;
    if (std::fclose(f)!=0) valid=false;
    return valid;
}
} // namespace vla
