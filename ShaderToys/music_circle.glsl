float waveData(float u){
    int tx = int(u * 512.0);
    float wave = texelFetch( iChannel0, ivec2(tx,1), 0 ).x;
    return wave;
}
#define pi 3.1415926535
#define pix2 6.28318531
#define LINE_WIDTH 40.0
#define N_SPLIT 32
#define SPAN_WIDTH (2.0*pi / float(N_SPLIT))
#define BLACK_OCC 0.3
float iR = 0.0;
float oR = 0.0;
bool radWhite(float rad){
    return mod(rad, SPAN_WIDTH) / SPAN_WIDTH > BLACK_OCC;
}
vec3 circle(float R, float rad){
    vec3 color = vec3(0.0);
    float u = floor(rad / SPAN_WIDTH) * SPAN_WIDTH;
    float wav = waveData(u / pix2)*2.1;
    wav = pow(wav, 8.0);
    float liR = iR, loR = oR + wav;
    if(R > liR){
        if(R < loR && radWhite(rad)){
            vec3 base = vec3(cos(rad + 2.0 + iTime), sin(rad + 3.0 + iTime), cos(rad + 4.0 + iTime)) + vec3(1.0);
            base = base / 2.0;
            color = base;
        }
    }else{
        color = vec3(0.0);
    }
    return color;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec3 color = vec3(0.0);
    vec2 xy = fragCoord - iResolution.xy / 2.0;
    float rad = atan(xy.y, xy.x) + pi;
    float r = length(xy);
    iR = float(min(iResolution.x, iResolution.y)) / 6.0;
    oR = iR + LINE_WIDTH;
    
    color = circle(r, rad);
    
    fragColor = vec4(color, 1.0);
}