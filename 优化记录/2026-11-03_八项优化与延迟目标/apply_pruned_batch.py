from pathlib import Path
p=Path('src/models/pi05.cpp');s=p.read_text()
a='''                GGML_ASSERT(nv == 1 && vt == GGML_TYPE_F32 && !vis.vit.enc.cfg.fused);'''
b='''                GGML_ASSERT(vt == GGML_TYPE_F32 && !vis.vit.enc.cfg.fused);'''
assert s.count(a)==1;s=s.replace(a,b)
a='''                        auto * scores = vision_patch_scores(VC, aw);
                        ggml_set_output(scores);
                        vio.scores.push_back(scores);'''
b='''                        for(int v=0;v<nv;++v) {
                            auto * view=nv==1?aw:ggml_view_3d(VC,aw,aw->ne[0],aw->ne[1],aw->ne[2],
                                                             aw->nb[1],aw->nb[2],v*aw->nb[3]);
                            auto * scores=vision_patch_scores(VC,view);
                            ggml_set_output(scores);vio.scores.push_back(scores);
                        }'''
assert s.count(a)==1;s=s.replace(a,b)
a='''            if (vision_keep && device_prune) {
                proj = vision_keep_rows(VC, proj, vio.scores.back(), (int)vision_keep);
                if (env_flag("VLA_PI05_DUMP_PRUNE")) ggml_set_output(proj->src[1]);
            }'''
b='''            if (vision_keep && device_prune) {
                ggml_tensor * last=nullptr;
                for(int v=0;v<nv;++v) {
                    auto * view=nv==1?proj:ggml_view_2d(VC,proj,proj->ne[0],K,proj->nb[1],
                                                       (size_t)v*K*proj->nb[1]);
                    last=vision_keep_rows(VC,view,vio.scores[vio.scores.size()-nv+v],(int)vision_keep);
                    if(env_flag("VLA_PI05_DUMP_PRUNE"))ggml_set_output(last->src[1]);
                    ggml_set_output(last);vio.emb.push_back(last);
                }
                return last;
            }'''
assert s.count(a)==1;s=s.replace(a,b)
a='''batch_views = batch_views || (musa_backend && !vision_keep && vis.vit.enc.cfg.gemm_act==GGML_TYPE_F32 && !env_flag("VLA_PI05_NO_VISION_BATCH"));'''
b='''batch_views = batch_views || (musa_backend && (!vision_keep || (device_prune && env_flag("VLA_PI05_PRUNED_BATCH"))) &&
            vis.vit.enc.cfg.gemm_act==GGML_TYPE_F32 && !env_flag("VLA_PI05_NO_VISION_BATCH"));'''
assert s.count(a)==1;s=s.replace(a,b)
a='''            ggml_build_forward_expand(vg, tower(new_px(n_images), n_images));'''
b='''            tower(new_px(n_images),n_images);'''
assert s.count(a)==1;s=s.replace(a,b)
a='''                ggml_build_forward_expand(vg, tower(new_px(1), 1));'''
b='''                tower(new_px(1),1);'''
assert s.count(a)==1;s=s.replace(a,b)
a='''        for (auto * scores : vio.scores) ggml_build_forward_expand(vg, scores);'''
b='''        for(auto * emb:vio.emb)ggml_build_forward_expand(vg,emb);
        for (auto * scores : vio.scores) ggml_build_forward_expand(vg, scores);'''
assert s.count(a)==1;s=s.replace(a,b);p.write_text(s)
p=Path('src/musa/vla_musa_fused.cu');s=p.read_text()
a='''    return a->ne[2] > 1 && a->ne[2] == b->ne[2] && a->ne[3] == 1 && b->ne[3] == 1 &&'''
b='''    return a->ne[2] > 1 && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3] &&'''
assert s.count(a)==1;s=s.replace(a,b)
a='''const dim3 grid((a->ne[1]+15)/16,(b->ne[1]+15)/16,a->ne[2]);'''
b='''const dim3 grid((a->ne[1]+15)/16,(b->ne[1]+15)/16,a->ne[2]*a->ne[3]);'''
assert s.count(a)==1;s=s.replace(a,b);p.write_text(s)
