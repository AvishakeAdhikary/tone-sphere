# MSVC has no C99 variable-length arrays; RNNoise v0.1.1 uses five. Rewrite them with alloca
# into patched copies of the two affected files (the fetched sources stay untouched).
# Inputs: SRC_DIR, OUT_DIR
file(MAKE_DIRECTORY ${OUT_DIR})

file(READ ${SRC_DIR}/src/celt_lpc.c lpc)
string(REPLACE "opus_val16 xx[n];" "opus_val16 *xx = (opus_val16*)alloca(sizeof(opus_val16)*(n));" lpc "${lpc}")
file(WRITE ${OUT_DIR}/celt_lpc.c "#include <malloc.h>\n${lpc}")

file(READ ${SRC_DIR}/src/pitch.c pitch)
string(REPLACE "opus_val16 x_lp4[len>>2];" "opus_val16 *x_lp4 = (opus_val16*)alloca(sizeof(opus_val16)*(len>>2));" pitch "${pitch}")
string(REPLACE "opus_val16 y_lp4[lag>>2];" "opus_val16 *y_lp4 = (opus_val16*)alloca(sizeof(opus_val16)*(lag>>2));" pitch "${pitch}")
string(REPLACE "opus_val32 xcorr[max_pitch>>1];" "opus_val32 *xcorr = (opus_val32*)alloca(sizeof(opus_val32)*(max_pitch>>1));" pitch "${pitch}")
string(REPLACE "opus_val32 yy_lookup[maxperiod+1];" "opus_val32 *yy_lookup = (opus_val32*)alloca(sizeof(opus_val32)*(maxperiod+1));" pitch "${pitch}")
file(WRITE ${OUT_DIR}/pitch.c "#include <malloc.h>\n${pitch}")
