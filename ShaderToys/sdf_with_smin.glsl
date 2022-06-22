// https://www.shadertoy.com/view/NllcR8
float sphereSDF(in vec3 samplePoint, in vec3 pos, in float radius){
   return length(samplePoint - pos) - radius;
}
float cubeSDF(in vec3 samplePoint, in vec3 pos, in vec3 b) {
  return length(max(abs(samplePoint - pos)-b,0.0));
}

float smin( float a, float b, float k){
     float h = max( k-abs(a-b), 0.0 )/k;
     return min(a, b) - h*h*k*(1.0/4.0);
}

float sceneSDF(in vec3 sp){
    float sp1 = sphereSDF(sp, vec3(2.0 * sin(iTime * 2.0), 0.0, 0), 0.7);
    float sp2 = sphereSDF(sp, vec3(0), 0.5);
    float cb1 = cubeSDF(sp, vec3(-2.0, 0, 0), vec3(0.5));
    return smin(smin(sp1, sp2, 0.9), cb1, 0.9);
}

#define DH 1e-4
vec3 sceneNormal(in vec3 sp){
    return normalize(vec3(
        sceneSDF(sp + vec3(DH, 0.0, 0.0)) - sceneSDF(sp),
        sceneSDF(sp + vec3(0.0, DH, 0.0)) - sceneSDF(sp),
        sceneSDF(sp + vec3(0.0, 0.0, DH)) - sceneSDF(sp)
    ));
}

struct inter{
    vec3 position;
    vec3 normal;
    vec3 refractDir;
};

float fresnelApprox(in vec3 I, in vec3 N){
    float R0 = (1.55 - 1.0) / (1.55 + 1.0);
    R0 *= R0;
    return R0 + (1.0 - R0) * pow(1.0 - (dot(I, N)), 5.0);
}

#define MAX_STEPS 100
#define EPS 1e-4
#define MAX_DIST 100.0
bool rayMarch(in vec3 ori, in vec3 dir, out inter hit){
    float stepSize = 0.0;
    vec3 p = ori;
    float depth = 0.0;
    for(int i = 0; i <= MAX_STEPS; i++){
        p += stepSize * dir;
        float dist = sceneSDF(p);
        depth += dist;
        if(dist < EPS){
            hit.position = p;
            hit.normal = sceneNormal(p);
            hit.refractDir = refract(dir, hit.normal, 0.66);
            return true;
        }
        if(depth > MAX_DIST){
            return false;
        }
        stepSize = dist;
    }
    return false;
}

vec3 rayDirection(float fovY, vec2 fragCoord){
    vec2 xy = fragCoord - iResolution.xy / 2.0;
    float z = iResolution.y / tan(radians(fovY) * 0.5);
    return normalize(vec3(xy, -z));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    
    vec3 ori = vec3(0.0, 0.0, 5.0);
    vec3 dir = rayDirection(90.0, fragCoord);
    
    inter hit;
    vec3 color = vec3(0.0);
    if(rayMarch(ori, dir, hit)){
        vec4 texColor;
        if(iMouse.z > 0.0){
            texColor = texture(iChannel0, hit.refractDir);
        }else{
            texColor = texture(iChannel0, reflect(dir, hit.normal));
        }
        color = texColor.rgb;
    }else{
        color = texture(iChannel0, dir).rgb;
    }
    // Output to screen
    fragColor = vec4(color, 1.0);
}