// origKernelName: _ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random27TruncatedNormalDistributionINS2_19SingleSampleAdapterINS2_12PhiloxRandomEEEfEEEEvS5_PNT_17ResultElementTypeExS8_
// uniqueKernelName: _ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random27TruncatedNormalDistributionINS2_19SingleSampleAdapterINS2_12PhiloxRandomEEEfEEEEvS5_PNT_17ResultElementTypeExS8__1_2_3
// shortKernelName: _ZN10tensorflow7func

// __vmem__ is just a marker, so we can see which bits are vmems
// It doesnt actually do anything; compiler ignores it
#define __vmem__

// vmem2 is a pointer to a pointer (so we have to unwrap twice)
#define __vmem2__

struct GlobalVars {
    local int *scratch;
    global char *clmem0;
    unsigned long clmem_vmem_offset0;
};

inline global float *getGlobalPointer(__vmem__ unsigned long vmemloc, const struct GlobalVars* const globalVars) {
    return (global float *)(globalVars->clmem0 + vmemloc - globalVars->clmem_vmem_offset0);
}

struct class_tensorflow__random__Array {
    int f0[4];
};
struct class_tensorflow__random__Array_0 {
    int f0[2];
};
struct class_tensorflow__random__TruncatedNormalDistribution {
    float f0;
};
struct class_tensorflow__random__Array_1 {
    float f0[4];
};
struct class_tensorflow__random__SingleSampleAdapter {
    global struct class_tensorflow__random__PhiloxRandom* f0;
    struct class_tensorflow__random__Array f1;
    int f2;
    char f3[4];
};
struct class_tensorflow__random__PhiloxRandom {
    struct class_tensorflow__random__Array f0;
    struct class_tensorflow__random__Array_0 f1;
};


inline unsigned int __umulhi(unsigned int v0, unsigned int v1) {
    unsigned long res = (unsigned long)v0 * v1;
    unsigned int res2 = res >> 32;
    return res2;
}
kernel void _ZN10tensorflow7func(global char* clmem0, unsigned long clmem_vmem_offset0, global char* clmem1, unsigned long clmem_vmem_offset1, global char* clmem2, unsigned long clmem_vmem_offset2, global char* clmem3, unsigned long clmem_vmem_offset3, long v30_offset, long v31_offset, long v32, long v33_offset, local int *scratch);

kernel void _ZN10tensorflow7func(global char* clmem0, unsigned long clmem_vmem_offset0, global char* clmem1, unsigned long clmem_vmem_offset1, global char* clmem2, unsigned long clmem_vmem_offset2, global char* clmem3, unsigned long clmem_vmem_offset3, long v30_offset, long v31_offset, long v32, long v33_offset, local int *scratch) {
    global struct class_tensorflow__random__TruncatedNormalDistribution* v33 = (global struct class_tensorflow__random__TruncatedNormalDistribution*)(clmem3 + v33_offset);
    global float* v31 = (global float*)(clmem2 + v31_offset);
    global struct class_tensorflow__random__PhiloxRandom* v30 = (global struct class_tensorflow__random__PhiloxRandom*)(clmem1 + v30_offset);

    const struct GlobalVars globalVars = { scratch, clmem0, clmem_vmem_offset0 };
    const struct GlobalVars* const pGlobalVars = &globalVars;

    char* v55;
    char* v63;
    char* v70;
    char* v75;
    char* v76;
    char* v77;
    char* v78;
    float v35[1];
    float v37[1];
    float v434;
    float v435;
    float v448;
    float v455;
    float v456;
    float v459;
    float v460;
    float v461;
    float v462;
    float v463;
    float v464;
    float v472;
    float v54;
    float* v469;
    float* v477;
    float* v71;
    float* v72;
    float* v73;
    float* v74;
    global float* v485;
    global float* v492;
    global float* v497;
    global float* v502;
    global int* v486;
    global int* v493;
    global int* v498;
    global int* v503;
    global struct class_tensorflow__random__PhiloxRandom** v64;
    int v103;
    int v104;
    int v108;
    int v109;
    int v110;
    int v113;
    int v114;
    int v119;
    int v120;
    int v136;
    int v137;
    int v139;
    int v140;
    int v141;
    int v142;
    int v144;
    int v147;
    int v150;
    int v153;
    int v155;
    int v157;
    int v163;
    int v165;
    int v167;
    int v169;
    int v175;
    int v177;
    int v179;
    int v181;
    int v187;
    int v189;
    int v191;
    int v193;
    int v199;
    int v201;
    int v203;
    int v205;
    int v211;
    int v213;
    int v215;
    int v217;
    int v223;
    int v225;
    int v227;
    int v229;
    int v235;
    int v237;
    int v239;
    int v241;
    int v247;
    int v249;
    int v251;
    int v253;
    int v258;
    int v259;
    int v260;
    int v261;
    int v263;
    int v265;
    int v266;
    int v267;
    int v270;
    int v271;
    int v272;
    int v273;
    int v274;
    int v275;
    int v276;
    int v277;
    int v280;
    int v281;
    int v284;
    int v285;
    int v287;
    int v291;
    int v293;
    int v294;
    int v295;
    int v296;
    int v297;
    int v298;
    int v300;
    int v302;
    int v304;
    int v306;
    int v308;
    int v310;
    int v314;
    int v316;
    int v318;
    int v320;
    int v324;
    int v326;
    int v328;
    int v330;
    int v334;
    int v336;
    int v338;
    int v340;
    int v344;
    int v346;
    int v348;
    int v34[1];
    int v350;
    int v354;
    int v356;
    int v358;
    int v360;
    int v364;
    int v366;
    int v368;
    int v36[1];
    int v370;
    int v374;
    int v376;
    int v378;
    int v380;
    int v384;
    int v386;
    int v388;
    int v390;
    int v393;
    int v394;
    int v395;
    int v396;
    int v398;
    int v400;
    int v401;
    int v402;
    int v405;
    int v406;
    int v409;
    int v410;
    int v413;
    int v414;
    int v41;
    int v421;
    int v425;
    int v429;
    int v42;
    int v444;
    int v44;
    int v466;
    int v467;
    int v46;
    int v474;
    int v475;
    int v47;
    int v480;
    int v481;
    int v482;
    int v483;
    int v98;
    int* v57;
    int* v58;
    int* v60;
    int* v61;
    int* v65;
    int* v66;
    int* v67;
    int* v68;
    int* v69;
    long v143;
    long v299;
    long v488;
    long v48;
    long v495;
    long v49;
    long v500;
    long v505;
    long v506;
    long v93;
    long v94;
    long* v92;
    struct class_tensorflow__random__Array_0* v91;
    struct class_tensorflow__random__Array_1 v38[1];
    struct class_tensorflow__random__PhiloxRandom v39[1];
    struct class_tensorflow__random__SingleSampleAdapter v40[1];

v1:;
    /* int v41 = call <unk> */;
    v41 = get_group_id(0);
    /* int v42 = call <unk> */;
    v42 = get_local_size(0);
    /* int v43 = mul v42 v41 */;
    /* int v44 = call <unk> */;
    v44 = get_local_id(0);
    /* int v45 = add v43 v44 */;
    /* int v46 = call <unk> */;
    v46 = get_num_groups(0);
    /* int v47 = mul v46 v42 */;
    v47 = v46 * v42;
    /* long v48 = sext v45 */;
    v48 = (long)((v42 * v41) + v44);
    /* long v49 = shl v48 <unk> */;
    v49 = v48 << 2;
    /* bool v51 = icmp v49 v32 */;
    /* if(v51) */
    if (v49 < v32) {
        goto v2;
    } else {
        goto v29;
    }
v2:;
    /* float* v52 = getelementptr v33 <unk> <unk> */;
    /* float v54 = load v52 */;
    v54 = (&(v33[0].f0))[0];
    /* char* v55 = bitcast v39 */;
    v55 = (char*)v39;
    /* char* v56 = bitcast v30 */;
    /* int* v57 = getelementptr v39 v53 <unk> <unk> v53 */;
    v57 = (&(v39[0].f0.f0[0]));
    /* int* v58 = getelementptr v39 v53 <unk> <unk> <unk> */;
    v58 = (&(v39[0].f0.f0[1]));
    /* int* v60 = getelementptr v39 v53 <unk> <unk> v50 */;
    v60 = (&(v39[0].f0.f0[2]));
    /* int* v61 = getelementptr v39 v53 <unk> <unk> <unk> */;
    v61 = (&(v39[0].f0.f0[3]));
    /* char* v63 = bitcast v40 */;
    v63 = (char*)v40;
    /* struct class_tensorflow__random__PhiloxRandom** v64 = getelementptr v40 v53 <unk> */;
    v64 = (&(v40[0].f0));
    /* int* v65 = getelementptr v40 v53 <unk> <unk> v53 */;
    v65 = (&(v40[0].f1.f0[0]));
    /* int* v66 = getelementptr v40 v53 <unk> <unk> v59 */;
    v66 = (&(v40[0].f1.f0[1]));
    /* int* v67 = getelementptr v40 v53 <unk> <unk> v50 */;
    v67 = (&(v40[0].f1.f0[2]));
    /* int* v68 = getelementptr v40 v53 <unk> <unk> v62 */;
    v68 = (&(v40[0].f1.f0[3]));
    /* int* v69 = getelementptr v40 v53 <unk> */;
    v69 = (&(v40[0].f2));
    /* char* v70 = bitcast v38 */;
    v70 = (char*)v38;
    /* float* v71 = getelementptr v38 v53 <unk> v53 */;
    v71 = (&(v38[0].f0[0]));
    /* float* v72 = getelementptr v38 v53 <unk> v59 */;
    v72 = (&(v38[0].f0[1]));
    /* float* v73 = getelementptr v38 v53 <unk> v50 */;
    v73 = (&(v38[0].f0[2]));
    /* float* v74 = getelementptr v38 v53 <unk> v62 */;
    v74 = (&(v38[0].f0[3]));
    /* char* v75 = bitcast v36 */;
    v75 = (char*)v36;
    /* char* v76 = bitcast v37 */;
    v76 = (char*)v37;
    /* char* v77 = bitcast v34 */;
    v77 = (char*)v34;
    /* char* v78 = bitcast v35 */;
    v78 = (char*)v35;
    /* int* v79 = bitcast v38 */;
    /* int* v80 = bitcast v72 */;
    /* int* v81 = bitcast v73 */;
    /* int* v82 = bitcast v74 */;
    /* int v83 = shl v47 <unk> */;
    /* int v85 = add v83 <unk> */;
    /* long v87 = sext v85 */;
    /* long v88 = sext v47 */;
    /* long v89 = add v87 <unk> */;
    /* struct class_tensorflow__random__Array_0* v91 = getelementptr v39 v53 <unk> */;
    v91 = (&(v39[0].f1));
    /* long* v92 = bitcast v91 */;
    v92 = (long*)v91;
    /* long v93 = phi v49 */
    v93 = v49;
    /* long v94 = phi v48 */
    v94 = v48;
    goto v3;
v3:;
    /* void v96 = call v55 v56 <unk> <unk> <unk> <unk> */;
    for(int __i=0; __i < 6; __i++) {;
        (( int *)v55)[__i] = ((global int *)((global char*)v30))[__i];
    }
;
    /* int v97 = trunc v94 */;
    /* int v98 = shl v97 <unk> */;
    v98 = (int)v94 << 8;
    /* long v100 = lshr v94 <unk> */;
    /* int v102 = trunc v100 */;
    /* int v103 = load v57 */;
    v103 = v57[0];
    /* int v104 = add v103 v98 */;
    v104 = v103 + v98;
    /* void v105 = store v104 v57 */;
    v57[0] = v104;
    /* bool v106 = icmp v104 v98 */;
    /* int v107 = zext v106 */;
    /* int v108 = add v107 v102 */;
    v108 = ((int)(v104 < v98)) + (int)(v94 >> 24);
    /* int v109 = load v58 */;
    v109 = v58[0];
    /* int v110 = add v108 v109 */;
    v110 = v108 + v109;
    /* void v111 = store v110 v58 */;
    v58[0] = v110;
    /* bool v112 = icmp v110 v108 */;
    /* if(v112) */
    if (v110 < v108) {
        goto v4;
    } else {
        goto v6;
    }
v4:;
    /* int v113 = load v60 */;
    v113 = v60[0];
    /* int v114 = add v113 <unk> */;
    v114 = v113 + 1;
    /* void v116 = store v114 v60 */;
    v60[0] = v114;
    /* bool v117 = icmp v114 <unk> */;
    /* if(v117) */
    if ((v114) == (0)) {
        goto v5;
    } else {
        goto v6;
    }
v5:;
    /* int v119 = load v61 */;
    v119 = v61[0];
    /* int v120 = add v119 v115 */;
    v120 = v119 + 1;
    /* void v121 = store v120 v61 */;
    v61[0] = v120;
    goto v6;
v6:;
    /* void v123 = store v39 v64 */;
    v64[0] = v39;
    /* void v124 = store v118 v65 */;
    v65[0] = 0;
    /* void v125 = store v118 v66 */;
    v66[0] = 0;
    /* void v126 = store v118 v67 */;
    v67[0] = 0;
    /* void v127 = store v118 v68 */;
    v68[0] = 0;
    /* void v128 = store <unk> v69 */;
    v69[0] = 4;
    /* void v131 = store <unk> v71 */;
    v71[0] = 0.0f;
    /* void v133 = store v132 v72 */;
    v72[0] = 0.0f;
    /* void v134 = store v132 v73 */;
    v73[0] = 0.0f;
    /* void v135 = store v132 v74 */;
    v74[0] = 0.0f;
    /* int v136 = phi v129 */
    v136 = 4;
    /* int v137 = phi v118 */
    v137 = 0;
    goto v7;
v7:;
    /* bool v138 = icmp v136 v129 */;
    /* if(v138) */
    if ((v136) == (4)) {
        goto v8;
    } else {
        goto v12;
    }
v8:;
    /* int v139 = load v57 */;
    v139 = v57[0];
    /* int v140 = load v58 */;
    v140 = v58[0];
    /* int v141 = load v60 */;
    v141 = v60[0];
    /* int v142 = load v61 */;
    v142 = v61[0];
    /* long v143 = load v92 */;
    v143 = v92[0];
    /* int v144 = trunc v143 */;
    v144 = (int)v143;
    /* long v145 = lshr v143 <unk> */;
    /* int v147 = trunc v145 */;
    v147 = (int)(v143 >> 32);
    /* int v148 = mul v139 <unk> */;
    /* int v150 = call v149 v139 <unk> */;
    v150 = __umulhi(-766435501, v139);
    /* int v151 = mul v141 <unk> */;
    /* int v153 = call v152 v141 <unk> */;
    v153 = __umulhi(-845247145, v141);
    /* int v154 = xor v144 v140 */;
    /* int v155 = xor v154 v153 */;
    v155 = (v144 ^ v140) ^ v153;
    /* int v156 = xor v150 v142 */;
    /* int v157 = xor v156 v147 */;
    v157 = (v150 ^ v142) ^ v147;
    /* int v158 = add v144 <unk> */;
    /* int v160 = add v147 <unk> */;
    /* int v162 = mul v155 v149 */;
    /* int v163 = call v149 v155 <unk> */;
    v163 = __umulhi(-766435501, v155);
    /* int v164 = mul v157 v152 */;
    /* int v165 = call v152 v157 <unk> */;
    v165 = __umulhi(-845247145, v157);
    /* int v166 = xor v158 v151 */;
    /* int v167 = xor v166 v165 */;
    v167 = ((v144 + -1640531527) ^ (v141 * -845247145)) ^ v165;
    /* int v168 = xor v163 v148 */;
    /* int v169 = xor v168 v160 */;
    v169 = (v163 ^ (v139 * -766435501)) ^ (v147 + -1150833019);
    /* int v170 = add v144 <unk> */;
    /* int v172 = add v147 <unk> */;
    /* int v174 = mul v167 v149 */;
    /* int v175 = call v149 v167 <unk> */;
    v175 = __umulhi(-766435501, v167);
    /* int v176 = mul v169 v152 */;
    /* int v177 = call v152 v169 <unk> */;
    v177 = __umulhi(-845247145, v169);
    /* int v178 = xor v164 v170 */;
    /* int v179 = xor v178 v177 */;
    v179 = ((v157 * -845247145) ^ (v144 + 1013904242)) ^ v177;
    /* int v180 = xor v162 v172 */;
    /* int v181 = xor v180 v175 */;
    v181 = ((v155 * -766435501) ^ (v147 + 1993301258)) ^ v175;
    /* int v182 = add v144 <unk> */;
    /* int v184 = add v147 <unk> */;
    /* int v186 = mul v179 v149 */;
    /* int v187 = call v149 v179 <unk> */;
    v187 = __umulhi(-766435501, v179);
    /* int v188 = mul v181 v152 */;
    /* int v189 = call v152 v181 <unk> */;
    v189 = __umulhi(-845247145, v181);
    /* int v190 = xor v176 v182 */;
    /* int v191 = xor v190 v189 */;
    v191 = ((v169 * -845247145) ^ (v144 + -626627285)) ^ v189;
    /* int v192 = xor v174 v184 */;
    /* int v193 = xor v192 v187 */;
    v193 = ((v167 * -766435501) ^ (v147 + 842468239)) ^ v187;
    /* int v194 = add v144 <unk> */;
    /* int v196 = add v147 <unk> */;
    /* int v198 = mul v191 v149 */;
    /* int v199 = call v149 v191 <unk> */;
    v199 = __umulhi(-766435501, v191);
    /* int v200 = mul v193 v152 */;
    /* int v201 = call v152 v193 <unk> */;
    v201 = __umulhi(-845247145, v193);
    /* int v202 = xor v188 v194 */;
    /* int v203 = xor v202 v201 */;
    v203 = ((v181 * -845247145) ^ (v144 + 2027808484)) ^ v201;
    /* int v204 = xor v186 v196 */;
    /* int v205 = xor v204 v199 */;
    v205 = ((v179 * -766435501) ^ (v147 + -308364780)) ^ v199;
    /* int v206 = add v144 <unk> */;
    /* int v208 = add v147 <unk> */;
    /* int v210 = mul v203 v149 */;
    /* int v211 = call v149 v203 <unk> */;
    v211 = __umulhi(-766435501, v203);
    /* int v212 = mul v205 v152 */;
    /* int v213 = call v152 v205 <unk> */;
    v213 = __umulhi(-845247145, v205);
    /* int v214 = xor v200 v206 */;
    /* int v215 = xor v214 v213 */;
    v215 = ((v193 * -845247145) ^ (v144 + 387276957)) ^ v213;
    /* int v216 = xor v198 v208 */;
    /* int v217 = xor v216 v211 */;
    v217 = ((v191 * -766435501) ^ (v147 + -1459197799)) ^ v211;
    /* int v218 = add v144 <unk> */;
    /* int v220 = add v147 <unk> */;
    /* int v222 = mul v215 v149 */;
    /* int v223 = call v149 v215 <unk> */;
    v223 = __umulhi(-766435501, v215);
    /* int v224 = mul v217 v152 */;
    /* int v225 = call v152 v217 <unk> */;
    v225 = __umulhi(-845247145, v217);
    /* int v226 = xor v212 v218 */;
    /* int v227 = xor v226 v225 */;
    v227 = ((v205 * -845247145) ^ (v144 + -1253254570)) ^ v225;
    /* int v228 = xor v210 v220 */;
    /* int v229 = xor v228 v223 */;
    v229 = ((v203 * -766435501) ^ (v147 + 1684936478)) ^ v223;
    /* int v230 = add v144 <unk> */;
    /* int v232 = add v147 <unk> */;
    /* int v234 = mul v227 v149 */;
    /* int v235 = call v149 v227 <unk> */;
    v235 = __umulhi(-766435501, v227);
    /* int v236 = mul v229 v152 */;
    /* int v237 = call v152 v229 <unk> */;
    v237 = __umulhi(-845247145, v229);
    /* int v238 = xor v224 v230 */;
    /* int v239 = xor v238 v237 */;
    v239 = ((v217 * -845247145) ^ (v144 + 1401181199)) ^ v237;
    /* int v240 = xor v222 v232 */;
    /* int v241 = xor v240 v235 */;
    v241 = ((v215 * -766435501) ^ (v147 + 534103459)) ^ v235;
    /* int v242 = add v144 <unk> */;
    /* int v244 = add v147 <unk> */;
    /* int v246 = mul v239 v149 */;
    /* int v247 = call v149 v239 <unk> */;
    v247 = __umulhi(-766435501, v239);
    /* int v248 = mul v241 v152 */;
    /* int v249 = call v152 v241 <unk> */;
    v249 = __umulhi(-845247145, v241);
    /* int v250 = xor v236 v242 */;
    /* int v251 = xor v250 v249 */;
    v251 = ((v229 * -845247145) ^ (v144 + -239350328)) ^ v249;
    /* int v252 = xor v234 v244 */;
    /* int v253 = xor v252 v247 */;
    v253 = ((v227 * -766435501) ^ (v147 + -616729560)) ^ v247;
    /* int v254 = add v144 <unk> */;
    /* int v256 = add v147 <unk> */;
    /* int v258 = mul v251 v149 */;
    v258 = v251 * -766435501;
    /* int v259 = call v149 v251 <unk> */;
    v259 = __umulhi(-766435501, v251);
    /* int v260 = mul v253 v152 */;
    v260 = v253 * -845247145;
    /* int v261 = call v152 v253 <unk> */;
    v261 = __umulhi(-845247145, v253);
    /* int v262 = xor v248 v254 */;
    /* int v263 = xor v262 v261 */;
    v263 = ((v241 * -845247145) ^ (v144 + -1879881855)) ^ v261;
    /* int v264 = xor v246 v256 */;
    /* int v265 = xor v264 v259 */;
    v265 = ((v239 * -766435501) ^ (v147 + -1767562579)) ^ v259;
    /* int v266 = load v57 */;
    v266 = v57[0];
    /* int v267 = add v266 v115 */;
    v267 = v266 + 1;
    /* void v268 = store v267 v57 */;
    v57[0] = v267;
    /* bool v269 = icmp v267 v118 */;
    /* if(v269) */
    if ((v267) == (0)) {
        goto v9;
    } else {
        /* int v270 = phi v263 */
        v270 = v263;
        /* int v271 = phi v260 */
        v271 = v260;
        /* int v272 = phi v265 */
        v272 = v265;
        /* int v273 = phi v258 */
        v273 = v258;
        /* int v274 = phi v115 */
        v274 = 1;
        /* int v275 = phi v263 */
        v275 = v263;
        goto v17;
    }
v9:;
    /* int v276 = load v58 */;
    v276 = v58[0];
    /* int v277 = add v276 v115 */;
    v277 = v276 + 1;
    /* void v278 = store v277 v58 */;
    v58[0] = v277;
    /* bool v279 = icmp v277 v118 */;
    /* if(v279) */
    if ((v277) == (0)) {
        goto v10;
    } else {
        /* int v270 = phi v263 */
        v270 = v263;
        /* int v271 = phi v260 */
        v271 = v260;
        /* int v272 = phi v265 */
        v272 = v265;
        /* int v273 = phi v258 */
        v273 = v258;
        /* int v274 = phi v115 */
        v274 = 1;
        /* int v275 = phi v263 */
        v275 = v263;
        goto v17;
    }
v10:;
    /* int v280 = load v60 */;
    v280 = v60[0];
    /* int v281 = add v280 v115 */;
    v281 = v280 + 1;
    /* void v282 = store v281 v60 */;
    v60[0] = v281;
    /* bool v283 = icmp v281 v118 */;
    /* if(v283) */
    if ((v281) == (0)) {
        goto v11;
    } else {
        /* int v270 = phi v263 */
        v270 = v263;
        /* int v271 = phi v260 */
        v271 = v260;
        /* int v272 = phi v265 */
        v272 = v265;
        /* int v273 = phi v258 */
        v273 = v258;
        /* int v274 = phi v115 */
        v274 = 1;
        /* int v275 = phi v263 */
        v275 = v263;
        goto v17;
    }
v11:;
    /* int v284 = load v61 */;
    v284 = v61[0];
    /* int v285 = add v284 v115 */;
    v285 = v284 + 1;
    /* void v286 = store v285 v61 */;
    v61[0] = v285;
    /* int v270 = phi v263 */
    v270 = v263;
    /* int v271 = phi v260 */
    v271 = v260;
    /* int v272 = phi v265 */
    v272 = v265;
    /* int v273 = phi v258 */
    v273 = v258;
    /* int v274 = phi v115 */
    v274 = 1;
    /* int v275 = phi v263 */
    v275 = v263;
    goto v17;
v12:;
    /* int v287 = add v136 v115 */;
    v287 = v136 + 1;
    /* void v288 = store v287 v69 */;
    v69[0] = v287;
    /* long v289 = sext v136 */;
    /* int* v290 = getelementptr v40 v53 v115 v118 v289 */;
    /* int v291 = load v290 */;
    v291 = (&(v40[0].f1.f0[(long)v136]))[0];
    /* bool v292 = icmp v287 v129 */;
    /* if(v292) */
    if ((v287) == (4)) {
        goto v13;
    } else {
        /* int v293 = phi v291 */
        v293 = v291;
        /* int v294 = phi v287 */
        v294 = v287;
        goto v18;
    }
v13:;
    /* int v295 = load v57 */;
    v295 = v57[0];
    /* int v296 = load v58 */;
    v296 = v58[0];
    /* int v297 = load v60 */;
    v297 = v60[0];
    /* int v298 = load v61 */;
    v298 = v61[0];
    /* long v299 = load v92 */;
    v299 = v92[0];
    /* int v300 = trunc v299 */;
    v300 = (int)v299;
    /* long v301 = lshr v299 v146 */;
    /* int v302 = trunc v301 */;
    v302 = (int)(v299 >> 32);
    /* int v303 = mul v295 v149 */;
    /* int v304 = call v149 v295 <unk> */;
    v304 = __umulhi(-766435501, v295);
    /* int v305 = mul v297 v152 */;
    /* int v306 = call v152 v297 <unk> */;
    v306 = __umulhi(-845247145, v297);
    /* int v307 = xor v300 v296 */;
    /* int v308 = xor v307 v306 */;
    v308 = (v300 ^ v296) ^ v306;
    /* int v309 = xor v304 v298 */;
    /* int v310 = xor v309 v302 */;
    v310 = (v304 ^ v298) ^ v302;
    /* int v311 = add v300 v159 */;
    /* int v312 = add v302 v161 */;
    /* int v313 = mul v308 v149 */;
    /* int v314 = call v149 v308 <unk> */;
    v314 = __umulhi(-766435501, v308);
    /* int v315 = mul v310 v152 */;
    /* int v316 = call v152 v310 <unk> */;
    v316 = __umulhi(-845247145, v310);
    /* int v317 = xor v311 v305 */;
    /* int v318 = xor v317 v316 */;
    v318 = ((v300 + -1640531527) ^ (v297 * -845247145)) ^ v316;
    /* int v319 = xor v314 v303 */;
    /* int v320 = xor v319 v312 */;
    v320 = (v314 ^ (v295 * -766435501)) ^ (v302 + -1150833019);
    /* int v321 = add v300 v171 */;
    /* int v322 = add v302 v173 */;
    /* int v323 = mul v318 v149 */;
    /* int v324 = call v149 v318 <unk> */;
    v324 = __umulhi(-766435501, v318);
    /* int v325 = mul v320 v152 */;
    /* int v326 = call v152 v320 <unk> */;
    v326 = __umulhi(-845247145, v320);
    /* int v327 = xor v315 v321 */;
    /* int v328 = xor v327 v326 */;
    v328 = ((v310 * -845247145) ^ (v300 + 1013904242)) ^ v326;
    /* int v329 = xor v313 v322 */;
    /* int v330 = xor v329 v324 */;
    v330 = ((v308 * -766435501) ^ (v302 + 1993301258)) ^ v324;
    /* int v331 = add v300 v183 */;
    /* int v332 = add v302 v185 */;
    /* int v333 = mul v328 v149 */;
    /* int v334 = call v149 v328 <unk> */;
    v334 = __umulhi(-766435501, v328);
    /* int v335 = mul v330 v152 */;
    /* int v336 = call v152 v330 <unk> */;
    v336 = __umulhi(-845247145, v330);
    /* int v337 = xor v325 v331 */;
    /* int v338 = xor v337 v336 */;
    v338 = ((v320 * -845247145) ^ (v300 + -626627285)) ^ v336;
    /* int v339 = xor v323 v332 */;
    /* int v340 = xor v339 v334 */;
    v340 = ((v318 * -766435501) ^ (v302 + 842468239)) ^ v334;
    /* int v341 = add v300 v195 */;
    /* int v342 = add v302 v197 */;
    /* int v343 = mul v338 v149 */;
    /* int v344 = call v149 v338 <unk> */;
    v344 = __umulhi(-766435501, v338);
    /* int v345 = mul v340 v152 */;
    /* int v346 = call v152 v340 <unk> */;
    v346 = __umulhi(-845247145, v340);
    /* int v347 = xor v335 v341 */;
    /* int v348 = xor v347 v346 */;
    v348 = ((v330 * -845247145) ^ (v300 + 2027808484)) ^ v346;
    /* int v349 = xor v333 v342 */;
    /* int v350 = xor v349 v344 */;
    v350 = ((v328 * -766435501) ^ (v302 + -308364780)) ^ v344;
    /* int v351 = add v300 v207 */;
    /* int v352 = add v302 v209 */;
    /* int v353 = mul v348 v149 */;
    /* int v354 = call v149 v348 <unk> */;
    v354 = __umulhi(-766435501, v348);
    /* int v355 = mul v350 v152 */;
    /* int v356 = call v152 v350 <unk> */;
    v356 = __umulhi(-845247145, v350);
    /* int v357 = xor v345 v351 */;
    /* int v358 = xor v357 v356 */;
    v358 = ((v340 * -845247145) ^ (v300 + 387276957)) ^ v356;
    /* int v359 = xor v343 v352 */;
    /* int v360 = xor v359 v354 */;
    v360 = ((v338 * -766435501) ^ (v302 + -1459197799)) ^ v354;
    /* int v361 = add v300 v219 */;
    /* int v362 = add v302 v221 */;
    /* int v363 = mul v358 v149 */;
    /* int v364 = call v149 v358 <unk> */;
    v364 = __umulhi(-766435501, v358);
    /* int v365 = mul v360 v152 */;
    /* int v366 = call v152 v360 <unk> */;
    v366 = __umulhi(-845247145, v360);
    /* int v367 = xor v355 v361 */;
    /* int v368 = xor v367 v366 */;
    v368 = ((v350 * -845247145) ^ (v300 + -1253254570)) ^ v366;
    /* int v369 = xor v353 v362 */;
    /* int v370 = xor v369 v364 */;
    v370 = ((v348 * -766435501) ^ (v302 + 1684936478)) ^ v364;
    /* int v371 = add v300 v231 */;
    /* int v372 = add v302 v233 */;
    /* int v373 = mul v368 v149 */;
    /* int v374 = call v149 v368 <unk> */;
    v374 = __umulhi(-766435501, v368);
    /* int v375 = mul v370 v152 */;
    /* int v376 = call v152 v370 <unk> */;
    v376 = __umulhi(-845247145, v370);
    /* int v377 = xor v365 v371 */;
    /* int v378 = xor v377 v376 */;
    v378 = ((v360 * -845247145) ^ (v300 + 1401181199)) ^ v376;
    /* int v379 = xor v363 v372 */;
    /* int v380 = xor v379 v374 */;
    v380 = ((v358 * -766435501) ^ (v302 + 534103459)) ^ v374;
    /* int v381 = add v300 v243 */;
    /* int v382 = add v302 v245 */;
    /* int v383 = mul v378 v149 */;
    /* int v384 = call v149 v378 <unk> */;
    v384 = __umulhi(-766435501, v378);
    /* int v385 = mul v380 v152 */;
    /* int v386 = call v152 v380 <unk> */;
    v386 = __umulhi(-845247145, v380);
    /* int v387 = xor v375 v381 */;
    /* int v388 = xor v387 v386 */;
    v388 = ((v370 * -845247145) ^ (v300 + -239350328)) ^ v386;
    /* int v389 = xor v373 v382 */;
    /* int v390 = xor v389 v384 */;
    v390 = ((v368 * -766435501) ^ (v302 + -616729560)) ^ v384;
    /* int v391 = add v300 v255 */;
    /* int v392 = add v302 v257 */;
    /* int v393 = mul v388 v149 */;
    v393 = v388 * -766435501;
    /* int v394 = call v149 v388 <unk> */;
    v394 = __umulhi(-766435501, v388);
    /* int v395 = mul v390 v152 */;
    v395 = v390 * -845247145;
    /* int v396 = call v152 v390 <unk> */;
    v396 = __umulhi(-845247145, v390);
    /* int v397 = xor v385 v391 */;
    /* int v398 = xor v397 v396 */;
    v398 = ((v380 * -845247145) ^ (v300 + -1879881855)) ^ v396;
    /* int v399 = xor v383 v392 */;
    /* int v400 = xor v399 v394 */;
    v400 = ((v378 * -766435501) ^ (v302 + -1767562579)) ^ v394;
    /* int v401 = load v57 */;
    v401 = v57[0];
    /* int v402 = add v401 v115 */;
    v402 = v401 + 1;
    /* void v403 = store v402 v57 */;
    v57[0] = v402;
    /* bool v404 = icmp v402 v118 */;
    /* if(v404) */
    if ((v402) == (0)) {
        goto v14;
    } else {
        /* int v270 = phi v398 */
        v270 = v398;
        /* int v271 = phi v395 */
        v271 = v395;
        /* int v272 = phi v400 */
        v272 = v400;
        /* int v273 = phi v393 */
        v273 = v393;
        /* int v274 = phi v118 */
        v274 = 0;
        /* int v275 = phi v291 */
        v275 = v291;
        goto v17;
    }
v14:;
    /* int v405 = load v58 */;
    v405 = v58[0];
    /* int v406 = add v405 v115 */;
    v406 = v405 + 1;
    /* void v407 = store v406 v58 */;
    v58[0] = v406;
    /* bool v408 = icmp v406 v118 */;
    /* if(v408) */
    if ((v406) == (0)) {
        goto v15;
    } else {
        /* int v270 = phi v398 */
        v270 = v398;
        /* int v271 = phi v395 */
        v271 = v395;
        /* int v272 = phi v400 */
        v272 = v400;
        /* int v273 = phi v393 */
        v273 = v393;
        /* int v274 = phi v118 */
        v274 = 0;
        /* int v275 = phi v291 */
        v275 = v291;
        goto v17;
    }
v15:;
    /* int v409 = load v60 */;
    v409 = v60[0];
    /* int v410 = add v409 v115 */;
    v410 = v409 + 1;
    /* void v411 = store v410 v60 */;
    v60[0] = v410;
    /* bool v412 = icmp v410 v118 */;
    /* if(v412) */
    if ((v410) == (0)) {
        goto v16;
    } else {
        /* int v270 = phi v398 */
        v270 = v398;
        /* int v271 = phi v395 */
        v271 = v395;
        /* int v272 = phi v400 */
        v272 = v400;
        /* int v273 = phi v393 */
        v273 = v393;
        /* int v274 = phi v118 */
        v274 = 0;
        /* int v275 = phi v291 */
        v275 = v291;
        goto v17;
    }
v16:;
    /* int v413 = load v61 */;
    v413 = v61[0];
    /* int v414 = add v413 v115 */;
    v414 = v413 + 1;
    /* void v415 = store v414 v61 */;
    v61[0] = v414;
    /* int v270 = phi v398 */
    v270 = v398;
    /* int v271 = phi v395 */
    v271 = v395;
    /* int v272 = phi v400 */
    v272 = v400;
    /* int v273 = phi v393 */
    v273 = v393;
    /* int v274 = phi v118 */
    v274 = 0;
    /* int v275 = phi v291 */
    v275 = v291;
    goto v17;
v17:;
    /* void v416 = store v270 v65 */;
    v65[0] = v270;
    /* void v417 = store v271 v66 */;
    v66[0] = v271;
    /* void v418 = store v272 v67 */;
    v67[0] = v272;
    /* void v419 = store v273 v68 */;
    v68[0] = v273;
    /* void v420 = store v274 v69 */;
    v69[0] = v274;
    /* int v293 = phi v275 */
    v293 = v275;
    /* int v294 = phi v274 */
    v294 = v274;
    goto v18;
v18:;
    /* int v421 = add v294 v115 */;
    v421 = v294 + 1;
    /* void v422 = store v421 v69 */;
    v69[0] = v421;
    /* long v423 = sext v294 */;
    /* int* v424 = getelementptr v40 v53 v115 v118 v423 */;
    /* int v425 = load v424 */;
    v425 = (&(v40[0].f1.f0[(long)v294]))[0];
    /* int v426 = and v293 <unk> */;
    /* int v429 = or v426 <unk> */;
    v429 = (v293 & 8388607) | 1065353216;
    /* void v431 = store v429 v36 */;
    v36[0] = v429;
    /* void v433 = call v76 v75 v90 <unk> */;
    (( int *)v76)[0] = (( int *)v75)[0];
    /* float v434 = load v37 */;
    v434 = v37[0];
    /* float v435 = fadd v434 <unk> */;
    v435 = v434 + -1.0f;
    /* bool v439 = fcmp v435 <unk> */;
    /* float v441 = select v439 v440 v435 */;
    /* int v442 = and v425 v427 */;
    /* int v444 = or v442 v430 */;
    v444 = (v425 & 8388607) | 1065353216;
    /* void v445 = store v444 v34 */;
    v34[0] = v444;
    /* void v447 = call v78 v77 v90 <unk> */;
    (( int *)v78)[0] = (( int *)v77)[0];
    /* float v448 = load v35 */;
    v448 = v35[0];
    /* float v449 = fadd v448 v436 */;
    /* float v452 = fpext v449 */;
    /* float v453 = fmul v452 <unk> */;
    /* float v455 = fptrunc v453 */;
    v455 = (float)(((float)(v448 + -1.0f)) * 6.28319f);
    /* float v456 = call v441 <unk> */;
    v456 = log((v435 < 1e-07f) ? 1e-07f : v435);
    /* float v457 = fmul v456 <unk> */;
    /* float v459 = call v457 <unk> */;
    v459 = sqrt(v456 * -2.0f);
    /* float v460 = call v455 <unk> */;
    v460 = sin(v455);
    /* float v461 = call v455 <unk> */;
    v461 = cos(v455);
    /* float v462 = fmul v459 v460 */;
    v462 = v459 * v460;
    /* float v463 = fmul v459 v461 */;
    v463 = v459 * v461;
    /* float v464 = call v462 <unk> */;
    v464 = fabs(v462);
    /* bool v465 = fcmp v464 v54 */;
    /* if(v465) */
    if (v464 < v54) {
        goto v19;
    } else {
        /* int v466 = phi v137 */
        v466 = v137;
        goto v20;
    }
v19:;
    /* int v467 = add v137 v115 */;
    v467 = v137 + 1;
    /* long v468 = sext v137 */;
    /* float* v469 = getelementptr v38 v53 v118 v468 */;
    v469 = (&(v38[0].f0[(long)v137]));
    /* void v470 = store v462 v469 */;
    v469[0] = v462;
    /* bool v471 = icmp v137 v84 */;
    /* if(v471) */
    if (v137 > 2) {
        goto v23;
    } else {
        /* int v466 = phi v467 */
        v466 = v467;
        goto v20;
    }
v20:;
    /* float v472 = call v463 <unk> */;
    v472 = fabs(v463);
    /* bool v473 = fcmp v472 v54 */;
    /* if(v473) */
    if (v472 < v54) {
        goto v21;
    } else {
        /* int v474 = phi v466 */
        v474 = v466;
        goto v22;
    }
v21:;
    /* int v475 = add v466 v115 */;
    v475 = v466 + 1;
    /* long v476 = sext v466 */;
    /* float* v477 = getelementptr v38 v53 v118 v476 */;
    v477 = (&(v38[0].f0[(long)v466]));
    /* void v478 = store v463 v477 */;
    v477[0] = v463;
    /* bool v479 = icmp v466 v84 */;
    /* if(v479) */
    if (v466 > 2) {
        goto v23;
    } else {
        /* int v474 = phi v475 */
        v474 = v475;
        goto v22;
    }
v22:;
    /* int v136 = phi v421 */
    v136 = v421;
    /* int v137 = phi v474 */
    v137 = v474;
    goto v7;
v23:;
    /* int v480 = load v79 */;
    v480 = ((int*)v38)[0];
    /* int v481 = load v80 */;
    v481 = ((int*)v72)[0];
    /* int v482 = load v81 */;
    v482 = ((int*)v73)[0];
    /* int v483 = load v82 */;
    v483 = ((int*)v74)[0];
    /* float* v485 = getelementptr v31 v93 */;
    v485 = (&(v31[v93]));
    /* int* v486 = bitcast v485 */;
    v486 = (global int*)v485;
    /* void v487 = store v480 v486 */;
    v486[0] = v480;
    /* long v488 = or v93 v59 */;
    v488 = v93 | 1;
    /* bool v489 = icmp v488 v32 */;
    /* if(v489) */
    if (v488 < v32) {
        goto v25;
    } else {
        goto v24;
    }
v24:;
    goto v29;
v25:;
    /* float* v492 = getelementptr v31 v488 */;
    v492 = (&(v31[v488]));
    /* int* v493 = bitcast v492 */;
    v493 = (global int*)v492;
    /* void v494 = store v481 v493 */;
    v493[0] = v481;
    /* long v495 = or v93 v50 */;
    v495 = v93 | 2;
    /* bool v496 = icmp v495 v32 */;
    /* if(v496) */
    if (v495 < v32) {
        goto v26;
    } else {
        goto v24;
    }
v26:;
    /* float* v497 = getelementptr v31 v495 */;
    v497 = (&(v31[v495]));
    /* int* v498 = bitcast v497 */;
    v498 = (global int*)v497;
    /* void v499 = store v482 v498 */;
    v498[0] = v482;
    /* long v500 = or v93 v62 */;
    v500 = v93 | 3;
    /* bool v501 = icmp v500 v32 */;
    /* if(v501) */
    if (v500 < v32) {
        goto v27;
    } else {
        goto v24;
    }
v27:;
    /* float* v502 = getelementptr v31 v500 */;
    v502 = (&(v31[v500]));
    /* int* v503 = bitcast v502 */;
    v503 = (global int*)v502;
    /* void v504 = store v483 v503 */;
    v503[0] = v483;
    /* long v505 = add v89 v93 */;
    v505 = (((long)((v47 << 2) + -4)) + 4) + v93;
    /* long v506 = add v94 v88 */;
    v506 = v94 + ((long)v47);
    /* bool v509 = icmp v505 v32 */;
    /* if(v509) */
    if (v505 < v32) {
        /* long v93 = phi v505 */
        v93 = v505;
        /* long v94 = phi v506 */
        v94 = v506;
        goto v3;
    } else {
        goto v28;
    }
v28:;
    goto v29;
v29:;
    return;
}

