#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <string>
#include <vector>
#include <iostream>
#include <map>

// STB IMAGE LIBRARY
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
constexpr int MAX_POINT_LIGHTS = 8; // Max point light sayısı
constexpr unsigned int SHADOW_WIDTH = 2048; 
constexpr unsigned int SHADOW_HEIGHT = 2048;
constexpr unsigned int DEFAULT_WINDOW_WIDTH = 1280;
constexpr unsigned int DEFAULT_WINDOW_HEIGHT = 720;

unsigned int WINDOW_WIDTH = DEFAULT_WINDOW_WIDTH;
unsigned int WINDOW_HEIGHT = DEFAULT_WINDOW_HEIGHT;


float fov = 45.0f; // Field of view for zoom
bool mouseCaptured = true; // Mouse capture state


struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
};

struct Material {
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 ambient;
    float shininess;
    bool hasTexture;
    GLuint diffuseMap;
    glm::vec3 emission;
    bool hasEmission;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material material;
    GLuint VAO, VBO, EBO;
};

struct Model {
    std::vector<Mesh> meshes;
    std::string directory;
    glm::vec3 position;
    glm::vec3 scale;
    glm::vec3 rotation; // for rotation around the Y-axis
    std::string name;

  
    bool useCustomTexture = false;
    GLuint customTextureID = 0;
    

    bool useBlendedTextures = false;
    GLuint blendedShaderID = 0;
    GLuint secondTextureID = 0;
    float blendFactor = 0.5f;

  
    bool useCustomShader = false;
    GLuint customShaderID = 0;
    

    bool isRotating = false;

    Model() : position(0.0f), scale(1.0f), rotation(0.0f), name("") {}

};

struct Terrain {
    GLuint VAO, VBO, EBO;
    GLuint textureID;
    unsigned int indexCount;
    glm::vec3 position;
    float scale;
    
    Terrain() : VAO(0), VBO(0), EBO(0), textureID(0), indexCount(0), position(0.0f), scale(1.0f) {}
};

struct PointLight {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
    float constant;
    float linear;
    float quadratic;
    bool isOn;

    
    PointLight() : position(0.0f), color(1.0f), intensity(1.0f), 
                   constant(1.0f), linear(0.09f), quadratic(0.032f), isOn(true) {}
    PointLight(const glm::vec3& pos, const glm::vec3& col, float intens) 
        : position(pos), color(col), intensity(intens), 
          constant(1.0f), linear(0.09f), quadratic(0.032f), isOn(true) {}               
};

// Lighting System Class
class LightingSystem {
    private:
        std::vector<PointLight> pointLights;
        glm::vec3 directionalLightDir;
        glm::vec3 directionalLightColor;
        float directionalLightIntensity;
        
    public:
        LightingSystem() : directionalLightDir(glm::normalize(glm::vec3(0.0f, -1.0f, 0.3f))),
                        directionalLightColor(0.7f, 0.8f, 1.0f),
                        directionalLightIntensity(0.5f) {}
        
       
        void addPointLight(const PointLight& light) {
            if (pointLights.size() < MAX_POINT_LIGHTS) {
                pointLights.push_back(light);
            }
        }
        
       
        void updatePointLight(size_t index, const PointLight& light) {
            if (index < pointLights.size()) {
                pointLights[index] = light;
            }
        }
        
  
        void removePointLight(size_t index) {
            if (index < pointLights.size()) {
                pointLights.erase(pointLights.begin() + index);
            }
        }
        
       
        void setDirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity) {
            directionalLightDir = glm::normalize(direction);
            directionalLightColor = color;
            directionalLightIntensity = intensity;
        }
        
     
        void sendToShader(GLuint shaderProgram, const glm::vec3& viewPos) const {
           
            glUniform3fv(glGetUniformLocation(shaderProgram, "lightDir"), 1, glm::value_ptr(directionalLightDir));
            glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(directionalLightColor));
            glUniform1f(glGetUniformLocation(shaderProgram, "lightIntensity"), directionalLightIntensity);
            glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(viewPos));
            
            
            glUniform1i(glGetUniformLocation(shaderProgram, "numPointLights"), static_cast<int>(pointLights.size()));
            
            for (size_t i = 0; i < pointLights.size() && i < MAX_POINT_LIGHTS; ++i) {
                std::string base = "pointLights[" + std::to_string(i) + "].";
                
                glUniform3fv(glGetUniformLocation(shaderProgram, (base + "position").c_str()), 1, glm::value_ptr(pointLights[i].position));
                glUniform3fv(glGetUniformLocation(shaderProgram, (base + "color").c_str()), 1, glm::value_ptr(pointLights[i].color));
                glUniform1f(glGetUniformLocation(shaderProgram, (base + "intensity").c_str()), pointLights[i].intensity);
                glUniform1f(glGetUniformLocation(shaderProgram, (base + "constant").c_str()), pointLights[i].constant);
                glUniform1f(glGetUniformLocation(shaderProgram, (base + "linear").c_str()), pointLights[i].linear);
                glUniform1f(glGetUniformLocation(shaderProgram, (base + "quadratic").c_str()), pointLights[i].quadratic);
                glUniform1i(glGetUniformLocation(shaderProgram, (base + "isOn").c_str()), pointLights[i].isOn);
            }
        }
        
     
        const std::vector<PointLight>& getPointLights() const { return pointLights; }
        const glm::vec3& getDirectionalLightDir() const { return directionalLightDir; }
        const glm::vec3& getDirectionalLightColor() const { return directionalLightColor; }
        float getDirectionalLightIntensity() const { return directionalLightIntensity; }
        size_t getPointLightCount() const { return pointLights.size(); }
};


std::map<std::string, Model> loadedModels; // Yüklenmiş modeller
std::vector<Model> sceneModels; // Sahnedeki model instanceları
Terrain terrain;
LightingSystem lightingSystem; 

GLuint shaderProgram;
glm::vec3 cameraPos = glm::vec3(0.0f, 5.0f, 10.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float deltaTime = 0.0f;
float frameCount = 0;
float lastTime = 0.0f;
float lastFrame = 0.0f;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = 400.0f;
float lastY = 300.0f;
bool firstMouse = true;


GLuint depthMapFBO;
GLuint depthMap;
GLuint shadowMapShaderProgram;
glm::mat4 lightSpaceMatrix;

// Axis-Aligned Bounding Box (AABB) yapısı
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    
    AABB() : min(0.0f), max(0.0f) {}
    AABB(const glm::vec3& minPoint, const glm::vec3& maxPoint) : min(minPoint), max(maxPoint) {}
    
    
    void updateFromModel(const glm::vec3& position, const glm::vec3& scale, const glm::vec3& baseSize) {
        glm::vec3 halfSize = (baseSize * scale) * 0.5f;
        min = position - halfSize;
        max = position + halfSize;
    }
    
    // Nokta AABB içinde mi?
    bool containsPoint(const glm::vec3& point) const {
        return (point.x >= min.x && point.x <= max.x &&
                point.y >= min.y && point.y <= max.y &&
                point.z >= min.z && point.z <= max.z);
    }
    
    // İki AABB çarpışıyor mu?
    bool intersects(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x &&
                min.y <= other.max.y && max.y >= other.min.y &&
                min.z <= other.max.z && max.z >= other.min.z);
    }
    
  
    AABB expanded(float padding) const {
        return AABB(min - glm::vec3(padding), max + glm::vec3(padding));
    }
};


struct CollisionObject {
    AABB boundingBox;
    std::string name;
    bool isStatic;
    bool isWall;
    
    CollisionObject() : isStatic(true), isWall(false) {}
    CollisionObject(const AABB& aabb, const std::string& objectName, bool static_obj = true, bool wall = false) 
        : boundingBox(aabb), name(objectName), isStatic(static_obj), isWall(wall) {}
};


class CollisionSystem {
    private:
       
        AABB worldBounds;
        float playerRadius;
        float playerHeight;
        
    public:
    std::vector<CollisionObject> staticObjects; 
        CollisionSystem() : playerRadius(0.3f), playerHeight(1.8f) { 

    // Dünya sınırlarını ayarla (terrain boyutuna göre)
    worldBounds = AABB(glm::vec3(-29.0f, 0.0f, -29.0f), glm::vec3(29.0f, 50.0f, 29.0f));
    }
        
    
        void addStaticObject(const glm::vec3& position, const glm::vec3& scale, const glm::vec3& baseSize, 
                            const std::string& name, bool isWall = false) {
            AABB aabb;
            aabb.updateFromModel(position, scale, baseSize);
            staticObjects.emplace_back(aabb, name, true, isWall);
        }
        
        // Statik objeleri AABB ile ekle
        void addCustomObject(const AABB& aabb, const std::string& name, bool isWall = false) {
            staticObjects.emplace_back(aabb, name, true, isWall);
        }
        
    
        bool isPositionValid(const glm::vec3& position) const {
       
            AABB playerBox(position - glm::vec3(playerRadius, 0.0f, playerRadius),
                          position + glm::vec3(playerRadius, playerHeight, playerRadius));
            
            // Dünya sınırları kontrolü (sadece X-Z kontrolü)
            if (position.x < worldBounds.min.x || position.x > worldBounds.max.x ||
                position.z < worldBounds.min.z || position.z > worldBounds.max.z) {
                return false;
            }
                
          
            for (const auto& obj : staticObjects) {
                if (playerBox.intersects(obj.boundingBox)) {
                    return false;
                }
            }
            
            return true;
        }
        
        glm::vec3 resolveMovement(const glm::vec3& currentPos, const glm::vec3& targetPos) const {

    glm::vec3 adjustedTarget = glm::vec3(targetPos.x, playerHeight, targetPos.z);
    glm::vec3 adjustedCurrent = glm::vec3(currentPos.x, playerHeight, currentPos.z);
    
   
    if (isPositionValid(adjustedTarget)) {
        return adjustedTarget;
    }
    

    glm::vec3 slideX = glm::vec3(adjustedTarget.x, playerHeight, adjustedCurrent.z);
    glm::vec3 slideZ = glm::vec3(adjustedCurrent.x, playerHeight, adjustedTarget.z);
    
  
    if (isPositionValid(slideX)) {
        return slideX;
    }
    

    if (isPositionValid(slideZ)) {
        return slideZ;
    }
    

    return adjustedCurrent;
    }
        

        
     
        void setPlayerRadius(float radius) { playerRadius = radius; }
        
       
        void setWorldBounds(const AABB& bounds) { worldBounds = bounds; }

        void setPlayerHeight(float height) { 
            playerHeight = height; 
        }
        
   
        void clear() { staticObjects.clear(); }
};


CollisionSystem collisionSystem;

// Shader kaynak kodları - Phong aydınlatma modeli
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;
    
    out vec2 TexCoords;
    out vec3 Normal;
    out vec3 FragPos;
    out vec4 FragPosLightSpace;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 lightSpaceMatrix;
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        TexCoords = aTexCoords;
        FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

// Fragment shader - Phong aydınlatma modeli
const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    in vec2 TexCoords;
    in vec3 Normal;
    in vec3 FragPos;
    in vec4 FragPosLightSpace; // Işık uzayındaki pozisyon
    
    #define MAX_POINT_LIGHTS 8

    struct PointLight {
        vec3 position;
        vec3 color;
        float intensity;
        float constant;
        float linear;
        float quadratic;
        bool isOn;
    };


    // Doku ve malzeme özellikleri
    uniform sampler2D diffuseMap;
    uniform vec3 material_diffuse;
    uniform vec3 material_specular;
    uniform vec3 material_ambient;
    uniform float material_shininess;
    uniform bool has_texture;

    uniform vec3 material_emission;
    uniform bool has_emission;

    // Shadow map
    uniform sampler2D shadowMap;
    
    // Directional light
    uniform vec3 lightDir;
    uniform vec3 lightColor;
    uniform float lightIntensity;

    // Point lights array
    uniform PointLight pointLights[MAX_POINT_LIGHTS];
    uniform int numPointLights;

    // Kamera pozisyonu
    uniform vec3 viewPos;
    
    // Gölge hesaplama fonksiyonu
    float ShadowCalculation(vec4 fragPosLightSpace) {
        // Perspektif bölme
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        
        // NDC koordinatlarını [0,1] aralığına dönüştür
        projCoords = projCoords * 0.5 + 0.5;
        
        // Geçerli koordinat kontrolü
        if(projCoords.z > 1.0)
            return 0.0;
            
        // En yakın derinlik değerini alın
        float closestDepth = texture(shadowMap, projCoords.xy).r;
        
        // Mevcut fragmanın derinliği
        float currentDepth = projCoords.z;
        
        // Bias hesapla - açıya göre bias değişir
        vec3 normal = normalize(Normal);
        vec3 lightDirection = normalize(-lightDir);
        float bias = max(0.05 * (1.0 - dot(normal, lightDirection)), 0.005);
        
        // PCF (Percentage Closer Filtering) - yumuşak gölgeler için
        float shadow = 0.0;
        vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
        for(int x = -1; x <= 1; ++x) {
            for(int y = -1; y <= 1; ++y) {
                float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
                shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
            }
        }
        shadow /= 9.0;
        
        return shadow;
    }

    // Tek Point light hesaplama fonksiyonu
    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 matDiffuse, vec3 matSpecular) {
        // Işık kapalıysa hiçbir şey döndürme
        if (!light.isOn) return vec3(0.0);
        
        vec3 lightDir = normalize(light.position - fragPos);
        
        // Diffuse
        float diff = max(dot(normal, lightDir), 0.0);
        
        // Specular
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material_shininess);
        
        // Attenuation (mesafeye göre ışık zayıflaması)
        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
        
        // Combine
        vec3 diffuse = light.color * diff * matDiffuse * light.intensity;
        vec3 specular = light.color * spec * matSpecular * light.intensity;
        
        diffuse *= attenuation;
        specular *= attenuation;
        
        return (diffuse + specular);
    }

    void main() {
        // Ambient aydınlatma
        vec3 ambient = material_ambient * lightColor * 0.3;
        
        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);
        
        // Directional light hesapla
        vec3 lightDirection = normalize(-lightDir);
        float diff = max(dot(norm, lightDirection), 0.0);
        vec3 directionalDiffuse = diff * lightColor * lightIntensity;
        
        vec3 reflectDir = reflect(-lightDirection, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material_shininess);
        vec3 directionalSpecular = spec * material_specular * lightColor * lightIntensity;

        // Gölge faktörünü hesapla
        float shadow = ShadowCalculation(FragPosLightSpace);
        
        // Tüm Point light'ları hesapla
        vec3 pointLightResult = vec3(0.0);
        vec3 material_diff_color = has_texture ? texture(diffuseMap, TexCoords).rgb : material_diffuse;
        
        for(int i = 0; i < numPointLights && i < MAX_POINT_LIGHTS; ++i) {
            pointLightResult += CalcPointLight(pointLights[i], norm, FragPos, viewDir, material_diff_color, material_specular);
        }
        
        // Malzeme rengini veya dokuyu kullan
        vec3 result;
        if (has_texture) {
            vec4 texColor = texture(diffuseMap, TexCoords);
            // Ambient her zaman, directional light gölgeden etkilenir, point light etkilenmez
            result = ambient * texColor.rgb + 
                     (1.0 - shadow) * (directionalDiffuse * texColor.rgb + directionalSpecular) + 
                     pointLightResult;

            // Emisyon varsa, emisyonu ekle
            if (has_emission) {
                result += material_emission;
            }
            FragColor = vec4(result, texColor.a);
        } else {
            result = ambient * material_diffuse + 
                     (1.0 - shadow) * (directionalDiffuse * material_diffuse + directionalSpecular) + 
                     pointLightResult;
            // Emisyon varsa, emisyonu ekle
            if (has_emission) {
                result += material_emission;
            }
            FragColor = vec4(result, 1.0);
        }
    }
)";

// Terrain vertex shader 
const char* terrainVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;
    
    out vec2 TexCoords;
    out vec3 Normal;
    out vec3 FragPos;
    out vec4 FragPosLightSpace;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 lightSpaceMatrix;
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        TexCoords = aTexCoords * 20.0; // Dokuyu tekrarlayarak daha detaylı görünüm
        FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const char* terrainFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    in vec2 TexCoords;
    in vec3 Normal;
    in vec3 FragPos;
    in vec4 FragPosLightSpace; // Işık uzayındaki pozisyon
    
    #define MAX_POINT_LIGHTS 8
    
    struct PointLight {
        vec3 position;
        vec3 color;
        float intensity;
        float constant;
        float linear;
        float quadratic;
        bool isOn;
    };

    uniform sampler2D terrainTexture;
    uniform sampler2D shadowMap;

    // Directional ışık
    uniform vec3 lightDir;
    uniform vec3 lightColor;
    uniform float lightIntensity;

    // Point lights array
    uniform PointLight pointLights[MAX_POINT_LIGHTS];
    uniform int numPointLights;

    // Kamera pozisyonu
    uniform vec3 viewPos;


    // ShadowCalculation fonksiyonunu burada tanımlamalısınız
    float ShadowCalculation(vec4 fragPosLightSpace) {
        // Perspektif bölme
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        
        // NDC koordinatlarını [0,1] aralığına dönüştür
        projCoords = projCoords * 0.5 + 0.5;
        
        // Geçerli koordinat kontrolü
        if(projCoords.z > 1.0)
            return 0.0;
            
        // En yakın derinlik değerini alın
        float closestDepth = texture(shadowMap, projCoords.xy).r;
        
        // Mevcut fragmanın derinliği
        float currentDepth = projCoords.z;
        
        // Bias hesapla - açıya göre bias değişir
        vec3 normal = normalize(Normal);
        vec3 lightDirection = normalize(-lightDir);
        float bias = max(0.05 * (1.0 - dot(normal, lightDirection)), 0.005);
        
        // PCF (Percentage Closer Filtering) - yumuşak gölgeler için
        float shadow = 0.0;
        vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
        for(int x = -1; x <= 1; ++x) {
            for(int y = -1; y <= 1; ++y) {
                float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
                shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
            }
        }
        shadow /= 9.0;
        
        return shadow;
    }

    // Point light hesaplama fonksiyonu - terrain için optimize edilmiş
    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 terrainColor) {
        // Işık kapalıysa hiçbir şey döndürme
        if (!light.isOn) return vec3(0.0);
        
        vec3 lightDir = normalize(light.position - fragPos);
        
        // Diffuse
        float diff = max(dot(normal, lightDir), 0.0);
        
        // Specular - terrain için daha az parlak
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        
        // Attenuation (mesafeye göre ışık zayıflaması)
        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
        
        // Combine
        vec3 diffuse = light.color * diff * terrainColor * light.intensity;
        vec3 specular = light.color * spec * 0.1 * light.intensity; // Terrain için daha az parlak
        
        diffuse *= attenuation;
        specular *= attenuation;
        
        return (diffuse + specular);
    }

    void main() {
        // Ambient aydınlatma
        vec3 ambient = vec3(0.3) * lightColor;
        
        // Diffuse aydınlatma
        vec3 norm = normalize(Normal);
        vec3 lightDirection = normalize(-lightDir);
        float diff = max(dot(norm, lightDirection), 0.0);
        vec3 diffuse = diff * lightColor * lightIntensity;
        
        // Specular aydınlatma
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDirection, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = spec * 0.1 * lightColor * lightIntensity; // Terrain çok parlak olmasın
        
        // Gölge hesapla
        float shadow = ShadowCalculation(FragPosLightSpace);
        
        vec4 texColor = texture(terrainTexture, TexCoords);
        
        // Tüm Point light'ları hesapla
        vec3 pointLightResult = vec3(0.0);
        for(int i = 0; i < numPointLights && i < MAX_POINT_LIGHTS; ++i) {
            pointLightResult += CalcPointLight(pointLights[i], norm, FragPos, viewDir, texColor.rgb);
        }
        pointLightResult *= 1.5; // Terrain'de point light'ları biraz daha belirgin yap
        
        // Tüm ışıkları birleştir - ambient ve directional gölgeden etkilenir, point light etkilenmez
        vec3 result = ambient * texColor.rgb + 
                      (1.0 - shadow) * (diffuse + specular) * texColor.rgb + 
                      pointLightResult;

        FragColor = vec4(result, 1.0);
    }
)";
const char* blendedTreeFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    in vec2 TexCoords;
    in vec3 Normal;
    in vec3 FragPos;
    in vec4 FragPosLightSpace;
    
    #define MAX_POINT_LIGHTS 8
    
    struct PointLight {
        vec3 position;
        vec3 color;
        float intensity;
        float constant;
        float linear;
        float quadratic;
        bool isOn;
    };
    
    // İki doku için sampler
    uniform sampler2D texture1; // Ağaç gövdesi dokusu
    uniform sampler2D texture2; // Yaprak dokusu
    uniform float blendFactor;  // Karıştırma faktörü (0.0 - 1.0)
    
    uniform sampler2D shadowMap;
    uniform vec3 material_diffuse;
    uniform vec3 material_specular;
    uniform vec3 material_ambient;
    uniform float material_shininess;
    
    // Directional ışık
    uniform vec3 lightDir;
    uniform vec3 lightColor;
    uniform float lightIntensity;
    
    // Point lights array
    uniform PointLight pointLights[MAX_POINT_LIGHTS];
    uniform int numPointLights;
    
    uniform vec3 viewPos;
    
    // Shadow hesaplama - FOR DÖNGÜSÜ DÜZELTİLDİ
    float ShadowCalculation(vec4 fragPosLightSpace) {
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords = projCoords * 0.5 + 0.5;
        
        if(projCoords.z > 1.0)
            return 0.0;
            
        float closestDepth = texture(shadowMap, projCoords.xy).r;
        float currentDepth = projCoords.z;
        
        vec3 normal = normalize(Normal);
        vec3 lightDirection = normalize(-lightDir);
        float bias = max(0.05 * (1.0 - dot(normal, lightDirection)), 0.005);
        
        // PCF - FOR DÖNGÜSÜ SYNTAX HATASI DÜZELTİLDİ
        float shadow = 0.0;
        vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
        
        // Manuel olarak 9 sample alın (3x3 grid)
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2(-1, -1) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2(-1,  0) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2(-1,  1) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2( 0, -1) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2( 0,  0) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2( 0,  1) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2( 1, -1) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2( 1,  0) * texelSize).r ? 1.0 : 0.0;
        shadow += currentDepth - bias > texture(shadowMap, projCoords.xy + vec2( 1,  1) * texelSize).r ? 1.0 : 0.0;
        
        shadow /= 9.0;
        return shadow;
    }
    
    // Point light hesaplama
    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 matDiffuse, vec3 matSpecular) {
        // Işık kapalıysa hiçbir şey döndürme
        if (!light.isOn) return vec3(0.0);
        
        vec3 lightDir = normalize(light.position - fragPos);
        float diff = max(dot(normal, lightDir), 0.0);
        
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material_shininess);
        
        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
        
        vec3 diffuse = light.color * diff * matDiffuse * light.intensity;
        vec3 specular = light.color * spec * matSpecular * light.intensity;
        
        diffuse *= attenuation;
        specular *= attenuation;
        
        return (diffuse + specular);
    }
    
    void main() {
        // İki dokuyu oku
        vec4 texColor1 = texture(texture1, TexCoords); // Bark
        vec4 texColor2 = texture(texture2, TexCoords); // Leaves
        
        // Dokuları karıştır - Y koordinatına göre
        float localBlendFactor = blendFactor * (0.5 + 0.5 * TexCoords.y);
        vec4 blendedColor = mix(texColor1, texColor2, localBlendFactor);

        // Aydınlatma hesaplamaları
        vec3 ambient = material_ambient * lightColor * 0.3;
        
        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);
        
        // Directional light
        vec3 lightDirection = normalize(-lightDir);
        float diff = max(dot(norm, lightDirection), 0.0);
        vec3 directionalDiffuse = diff * lightColor * lightIntensity;
        
        vec3 reflectDir = reflect(-lightDirection, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material_shininess);
        vec3 directionalSpecular = spec * material_specular * lightColor * lightIntensity;
        
        // Shadow
        float shadow = ShadowCalculation(FragPosLightSpace);
        
        // Tüm Point light'ları hesapla
        vec3 pointLightResult = vec3(0.0);
        for(int i = 0; i < numPointLights && i < MAX_POINT_LIGHTS; ++i) {
            pointLightResult += CalcPointLight(pointLights[i], norm, FragPos, viewDir, blendedColor.rgb, material_specular);
        }
        
        // Final result
        vec3 result = ambient * blendedColor.rgb + 
                      (1.0 - shadow) * (directionalDiffuse * blendedColor.rgb + directionalSpecular) + 
                      pointLightResult;
        
        FragColor = vec4(result, blendedColor.a);
    }
)";
const char* shadowMapVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 lightSpaceMatrix;
    uniform mat4 model;
    
    void main() {
        gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
    }
)";
const char* shadowMapFragmentShaderSource = R"(
    #version 330 core
    
    void main() {
        // Boş fragment shader - derinlik değerleri otomatik yazılır
    }
)";
const char* lampVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoord;

    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoord;
    out vec3 WorldPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 lightSpaceMatrix;

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        TexCoord = aTexCoord;
        WorldPos = aPos; // Original vertex position
        
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const char* lampFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoord;
    in vec3 WorldPos;

    #define MAX_POINT_LIGHTS 8
    
    struct PointLight {
        vec3 position;
        vec3 color;
        float intensity;
        float constant;
        float linear;
        float quadratic;
        bool isOn;
    };

    uniform sampler2D glassTexture;
    uniform sampler2D shadowMap;
    uniform mat4 lightSpaceMatrix;

    // Lighting uniforms
    uniform vec3 lightDir;
    uniform vec3 lightColor;
    uniform float lightIntensity;
    uniform vec3 viewPos;

    // Point lights array
    uniform PointLight pointLights[MAX_POINT_LIGHTS];
    uniform int numPointLights;

    float ShadowCalculation(vec4 fragPosLightSpace) {
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords = projCoords * 0.5 + 0.5;
        
        float closestDepth = texture(shadowMap, projCoords.xy).r;
        float currentDepth = projCoords.z;
        
        float bias = 0.005;
        float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
        
        return shadow;
    }

    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 materialColor) {
        // Işık kapalıysa hiçbir şey döndürme
        if (!light.isOn) return vec3(0.0);
        
        vec3 lightDirection = light.position - fragPos;
        float distance = length(lightDirection);
        lightDirection = normalize(lightDirection);
        
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
        float diff = max(dot(normal, lightDirection), 0.0);
        return diff * light.color * light.intensity * attenuation * materialColor;
    }

    void main() {
        vec3 norm = normalize(Normal);
        vec3 lightDirection = normalize(-lightDir);
        
        // Y koordinatına göre cam/direk ayırma
        // Model space'deki Y koordinatı 9.5'ten büyükse cam kısmı
        float glassThreshold = 9.5;
        
        vec3 finalColor;
        
        if (WorldPos.y > glassThreshold) {
            // CAM KISMI - Texture kullan
            vec4 glassColor = texture(glassTexture, TexCoord);
            
            // Directional light
            float diff = max(dot(norm, lightDirection), 0.0);
            vec3 diffuse = diff * lightColor * lightIntensity;
            vec3 ambient = 0.2 * lightColor;
            
            // Tüm Point light'ları hesapla
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 pointLightContrib = vec3(0.0);
            for(int i = 0; i < numPointLights && i < MAX_POINT_LIGHTS; ++i) {
                pointLightContrib += CalcPointLight(pointLights[i], norm, FragPos, viewDir, glassColor.rgb);
            }
            
            vec3 emission = glassColor.rgb * 0.8;
            
            // Shadow hesapla
            vec4 fragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
            float shadow = ShadowCalculation(fragPosLightSpace);
            
            finalColor = (ambient + (1.0 - shadow) * diffuse + pointLightContrib) * glassColor.rgb + emission;
            FragColor = vec4(finalColor, glassColor.a);
            
        } else {
            // DİREK KISMI - Gri renk
            vec3 grayColor = vec3(0.4, 0.4, 0.4);
            
            // Directional light
            float diff = max(dot(norm, lightDirection), 0.0);
            vec3 diffuse = diff * lightColor * lightIntensity;
            vec3 ambient = 0.2 * lightColor;
            
            // Tüm Point light'ları hesapla
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 pointLightContrib = vec3(0.0);
            for(int i = 0; i < numPointLights && i < MAX_POINT_LIGHTS; ++i) {
                pointLightContrib += CalcPointLight(pointLights[i], norm, FragPos, viewDir, grayColor);
            }
            
            // Shadow hesapla
            vec4 fragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
            float shadow = ShadowCalculation(fragPosLightSpace);
            
            finalColor = (ambient + (1.0 - shadow) * diffuse + pointLightContrib) * grayColor;
            FragColor = vec4(finalColor, 1.0);
        }
    }
)";


bool gateIsOpen = false; 
glm::vec3 gateClosedPos; 
glm::vec3 gateOpenPos;   
glm::vec3 gateCurrentPos; 
float gateAnimationSpeed = 2.0f; 
bool gateIsAnimating = false; 
int gateModelIndex = -1; 

float gateAnimationProgress = 0.0f;

float gateWidth = 5.0f; 
float gateDepth = 0.2f;  
glm::vec3 gateControlPostPos;
int gateControlPostIndex = 3; 

// Smooth interpolation 
float easeInOutQuad(float t) {
    if (t < 0.5f) {
        return 2.0f * t * t;
    } else {
        return -1.0f + (4.0f - 2.0f * t) * t;
    }
}

glm::mat4 calculateGateMatrix(float rotationAngle, const glm::vec3& hingePos, const glm::vec3& scale) {
    glm::mat4 matrix = glm::mat4(1.0f);
    matrix = glm::translate(matrix, hingePos);
    matrix = glm::rotate(matrix, glm::radians(rotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
    matrix = glm::translate(matrix, glm::vec3(gateWidth * 0.5f, 0.0f, 0.0f));
    matrix = glm::scale(matrix, scale);
    return matrix;
}

void findGateModelIndex() {
    for (size_t i = 0; i < sceneModels.size(); i++) {
        // Gate modelini name ile değil, setupFencingSystem'da eklenen sırayla bulalım
        // veya name'i kontrol edelim
        if (sceneModels[i].name == "gate" || 
            sceneModels[i].name.find("gate") != std::string::npos) {
            gateModelIndex = i;
         
            return;
        }
    }

    std::cout << "ERROR: Gate model not found!" << std::endl;
}


bool isPlayerNearGateControlPost() {
    float distance = glm::distance(glm::vec2(cameraPos.x, cameraPos.z), 
                                   glm::vec2(gateControlPostPos.x, gateControlPostPos.z));
    
    float proximityDistance = 2.5f; 
    return distance < proximityDistance;
}

void removeGateCollision() {
    auto it = collisionSystem.staticObjects.begin();
    while (it != collisionSystem.staticObjects.end()) {
        if (it->name == "Gate_Closed" || it->name == "Gate_Open") {
            
            it = collisionSystem.staticObjects.erase(it);
        } else {
            ++it;
        }
    }
}


void addGateClosedCollision() {
    glm::vec3 currentGatePosition = gateClosedPos; 

    float fenceCollisionWidth = 5.0f;
    float fenceCollisionHeight = 2.0f;
    float fenceCollisionDepth = 0.3f;
    
    AABB gateAABB(currentGatePosition - glm::vec3(fenceCollisionWidth/2, 0.0f, fenceCollisionDepth/2),
                  currentGatePosition + glm::vec3(fenceCollisionWidth/2, fenceCollisionHeight, fenceCollisionDepth/2));
    
    collisionSystem.addCustomObject(gateAABB, "Gate_Closed", true);
   
}


void addGateOpenCollision() {
    glm::vec3 currentGatePosition = gateOpenPos; 

    float fenceCollisionWidth = 5.0f;
    float fenceCollisionHeight = 2.0f;
    float fenceCollisionDepth = 0.3f;

    AABB gateAABB(currentGatePosition - glm::vec3(fenceCollisionDepth/2, 0.0f, fenceCollisionWidth/2),
                  currentGatePosition + glm::vec3(fenceCollisionDepth/2, fenceCollisionHeight, fenceCollisionWidth/2));
    
    collisionSystem.addCustomObject(gateAABB, "Gate_Open", true);
   
}


void addGateCollision() {
    if (gateIsOpen) {
      
        addGateOpenCollision();
    } else {

        addGateClosedCollision();
    }
}

void toggleGate() {
    if (gateIsAnimating) return; 
    
    if (!isPlayerNearGateControlPost()) {
        std::cout << "You need to be closer to the gate to open/close it!" << std::endl;
        return;
    }
    
   
    if (gateModelIndex == -1) {
        findGateModelIndex();
        if (gateModelIndex == -1) return;
    }

    gateIsOpen = !gateIsOpen;
    gateIsAnimating = true;
    
    removeGateCollision();

    if (gateIsOpen) {
        std::cout << "Opening gate..." << std::endl;

    } else {
        std::cout << "Closing gate..." << std::endl;

    }
}

void updateGateAnimation(float deltaTime) {
    if (!gateIsAnimating) return;
    
    if (gateModelIndex == -1) {
        findGateModelIndex();
        if (gateModelIndex == -1) return;
    }
    
    // Progress'i güncelle
    float deltaProgress = gateAnimationSpeed * deltaTime;
    
    if (gateIsOpen) {
        // Açılma: 0 -> 1
        gateAnimationProgress += deltaProgress;
        if (gateAnimationProgress >= 1.0f) {
            gateAnimationProgress = 1.0f;
            gateIsAnimating = false;
        }
    } else {
        // Kapanma: 1 -> 0  
        gateAnimationProgress -= deltaProgress;
        if (gateAnimationProgress <= 0.0f) {
            gateAnimationProgress = 0.0f;
            gateIsAnimating = false;
        }
    }
    
    // Smooth easing uygula
    float easedProgress = easeInOutQuad(gateAnimationProgress);
    
    // Pozisyon interpolasyonu - SMOOTH
    glm::vec3 targetPos = glm::mix(gateClosedPos, gateOpenPos, easedProgress);
    
    // Rotasyon interpolasyonu - SMOOTH  
    float targetRotationY = glm::mix(0.0f, 90.0f, easedProgress);
    glm::vec3 targetRotation = glm::vec3(0.0f, targetRotationY, 0.0f);
    
    // Model güncelleme
    sceneModels[gateModelIndex].position = targetPos;
    sceneModels[gateModelIndex].rotation = targetRotation;
    gateCurrentPos = targetPos;
    
    
    // Animasyon tamamlandığında
    if (!gateIsAnimating) {
        addGateCollision();
    }
}



enum CameraMode {
    PLAYER_MODE,  // First-person
    GOD_MODE      // Kuş bakışı
};


CameraMode currentCameraMode = PLAYER_MODE; 
glm::vec3 playerPosition = glm::vec3(0.0f, 1.8f, 15.0f); 
glm::vec3 godPosition = glm::vec3(0.0f, 30.0f, 0.0f);   
bool modeJustChanged = false;                        
float playerHeight = 1.8f;                               
float playerSpeed = 5.0f;                               
float godSpeed = 20.0f;                                  





void setupShadowMap() {
    
    glGenFramebuffers(1, &depthMapFBO);
    
  
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
                 SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    
   
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

 
}


glm::mat4 calculateRotatingTreeMatrix(float time, const glm::vec3& position, const glm::vec3& scale) {
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    float rotationAngle = glm::radians(time * 15.0f); 
    model = glm::rotate(model, rotationAngle, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::scale(model, scale);
    return model;
}



void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!mouseCaptured) return; 
    
    if (modeJustChanged) {
        lastX = xpos;
        lastY = ypos;
        modeJustChanged = false;
        return;
    }
    
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; 
    lastX = xpos;
    lastY = ypos;
    
    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    
    yaw += xoffset;
    pitch += yoffset;
    
    if (currentCameraMode == PLAYER_MODE) {
        if (pitch > 80.0f) pitch = 80.0f;
    } else {
       
        if (pitch > 89.0f) pitch = 89.0f;
    }
    if (pitch < -89.0f) pitch = -89.0f;
    
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {

  

   
    fov -= (float)yoffset * 2.0f; 
    
 
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 120.0f) fov = 120.0f;
    
    std::cout << "FOV: " << fov << std::endl;
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mouseCaptured = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        std::cout << "Mouse captured" << std::endl;
    }
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
        return;
    }
    static bool tabPressed = false;
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS && !tabPressed) {
        tabPressed = true;
        mouseCaptured = !mouseCaptured;
        if (mouseCaptured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            std::cout << "Mouse captured" << std::endl;
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            std::cout << "Mouse released" << std::endl;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_RELEASE) {
        tabPressed = false;
    }

    float cameraSpeed = (currentCameraMode == PLAYER_MODE) ? playerSpeed * deltaTime : godSpeed * deltaTime;

    glm::vec3 oldPos = cameraPos;

    if (currentCameraMode == PLAYER_MODE) {
        glm::vec3 forward = cameraFront;
        forward.y = 0.0f; 
        forward = glm::normalize(forward);   
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 newPos = oldPos;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            newPos += forward * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            newPos -= forward * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            newPos -= right * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            newPos += right * cameraSpeed;

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            newPos += forward * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            newPos -= forward * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            newPos -= right * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            newPos += right * cameraSpeed;

        static bool wasNearControlPost = false;
        bool isNearControlPost = isPlayerNearGateControlPost();     
        wasNearControlPost = isNearControlPost;     
        newPos.y = playerHeight; 
        cameraPos = collisionSystem.resolveMovement(oldPos, newPos);
        
    } else {
        glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= right * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += right * cameraSpeed;

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            cameraPos -= right * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            cameraPos += right * cameraSpeed;
 
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraUp;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraUp;
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

    if (key == GLFW_KEY_N && action == GLFW_PRESS) {
        if (currentCameraMode == PLAYER_MODE) {
            currentCameraMode = GOD_MODE;

            cameraPos = glm::vec3(cameraPos.x, godPosition.y, cameraPos.z);
            pitch = -45.0f; 

            glm::vec3 front;
            front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            front.y = sin(glm::radians(pitch));
            front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            cameraFront = glm::normalize(front);
            
        } else {
            currentCameraMode = PLAYER_MODE;
            
         
            cameraPos = glm::vec3(cameraPos.x, playerHeight, cameraPos.z);
            pitch = 0.0f; 

            glm::vec3 front;
            front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            front.y = sin(glm::radians(pitch));
            front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            cameraFront = glm::normalize(front);

        }
        modeJustChanged = true;
    }

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    if (key == GLFW_KEY_E && action == GLFW_PRESS) {
        toggleGate();
    }
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        gateAnimationSpeed = (gateAnimationSpeed == 2.5f) ? 1.0f : 2.5f;
    }

    
    // R tuşu - FOV reset
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        fov = 45.0f;
    }
}

void setupInputCallbacks(GLFWwindow* window) {

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetKeyCallback(window, key_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    mouseCaptured = true;

}


glm::mat4 getProjectionMatrix(float aspectRatio) {
    return glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 100.0f);
}

GLuint createShader(const char* vertexSource, const char* fragmentSource) {
   
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }

    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

 
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

GLuint createDefaultTexture() {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    unsigned char data[] = {255, 255, 255, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    return textureID;
}

GLuint loadTexture(const char* path) {
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrChannels;
    unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);
    
    if (data) {
        GLenum format;
        if (nrChannels == 1)
            format = GL_RED;
        else if (nrChannels == 3)
            format = GL_RGB;
        else if (nrChannels == 4)
            format = GL_RGBA;
        else
            format = GL_RGB; 
        
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
        return textureID;
    } else {
        std::cerr << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
        return createDefaultTexture();
    }
}

void loadMaterialProperties(aiMaterial* material, Material& outMaterial, const std::string& directory) {

    aiString matName;


    outMaterial.diffuse = glm::vec3(0.8f, 0.8f, 0.8f);  
    outMaterial.specular = glm::vec3(0.5f, 0.5f, 0.5f); 
    outMaterial.ambient = glm::vec3(0.2f, 0.2f, 0.2f);  
    outMaterial.shininess = 32.0f;                     
    outMaterial.hasTexture = false;

    aiColor3D diffuse;
    if (material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS) {
        outMaterial.diffuse = glm::vec3(diffuse.r, diffuse.g, diffuse.b);
    }
    else {
        std::cout << "  Failed to load diffuse color!" << std::endl;
    }
    
    
    aiColor3D specular;
    if (material->Get(AI_MATKEY_COLOR_SPECULAR, specular) == AI_SUCCESS) {
        if (specular.r > 0.01f || specular.g > 0.01f || specular.b > 0.01f) {
            outMaterial.specular = glm::vec3(specular.r, specular.g, specular.b);
        }
    }
    else {
        std::cout << "  Failed to load specular color!" << std::endl;
    }
    
    aiColor3D ambient;
    if (material->Get(AI_MATKEY_COLOR_AMBIENT, ambient) == AI_SUCCESS) {
        if (ambient.r > 0.01f || ambient.g > 0.01f || ambient.b > 0.01f) {
            outMaterial.ambient = glm::vec3(ambient.r, ambient.g, ambient.b);
        } else {
            outMaterial.ambient = outMaterial.diffuse * 0.3f;
        }
    }
    else {
        outMaterial.ambient = outMaterial.diffuse * 0.3f;
    }
    
    float shininess;
    if (material->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
        if (shininess > 0.1f) {
            outMaterial.shininess = shininess;
        }
    }
    else {
        std::cout << "  Failed to load shininess!" << std::endl;
    }

    aiString texPath;
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
        std::string fullPath = directory + "/" + texPath.C_Str();
        std::cout << "Loading texture: " << fullPath << std::endl;
        outMaterial.diffuseMap = loadTexture(fullPath.c_str());
        outMaterial.hasTexture = true;
    } else {
        outMaterial.diffuseMap = createDefaultTexture();
        outMaterial.hasTexture = false;
    }
}

Mesh processMesh(aiMesh* mesh, const aiScene* scene, const std::string& directory) {
    Mesh newMesh;

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;

        vertex.position.x = mesh->mVertices[i].x;
        vertex.position.y = mesh->mVertices[i].y;
        vertex.position.z = mesh->mVertices[i].z;

        if (mesh->HasNormals()) {
            vertex.normal.x = mesh->mNormals[i].x;
            vertex.normal.y = mesh->mNormals[i].y;
            vertex.normal.z = mesh->mNormals[i].z;
        }

        if (mesh->mTextureCoords[0]) {
            vertex.texCoords.x = mesh->mTextureCoords[0][i].x;
            vertex.texCoords.y = mesh->mTextureCoords[0][i].y;
        } else {
            vertex.texCoords = glm::vec2(0.0f, 0.0f);
        }

        newMesh.vertices.push_back(vertex);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            newMesh.indices.push_back(face.mIndices[j]);
    }

    if (mesh->mMaterialIndex >= 0) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        loadMaterialProperties(material, newMesh.material, directory);
    }

    glGenVertexArrays(1, &newMesh.VAO);
    glGenBuffers(1, &newMesh.VBO);
    glGenBuffers(1, &newMesh.EBO);
    
    glBindVertexArray(newMesh.VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, newMesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, newMesh.vertices.size() * sizeof(Vertex), &newMesh.vertices[0], GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, newMesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, newMesh.indices.size() * sizeof(unsigned int), &newMesh.indices[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
  
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    
    glBindVertexArray(0);
    
    return newMesh;
}

void processNode(aiNode* node, const aiScene* scene, Model& model) {

    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        model.meshes.push_back(processMesh(mesh, scene, model.directory));
    }
    
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, model);
    }
}

bool loadModel(const std::string& path, const std::string& modelName) {
    if (loadedModels.find(modelName) != loadedModels.end()) {
        return true;
    }
    Model model;
    size_t lastSlash = path.find_last_of("/\\");
    model.directory = lastSlash != std::string::npos ? path.substr(0, lastSlash) : ".";
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, 
        aiProcess_Triangulate | 
        aiProcess_FlipUVs | 
        aiProcess_CalcTangentSpace | 
        aiProcess_GenNormals);
    
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return false;
    }

    processNode(scene->mRootNode, scene, model);
    model.position = glm::vec3(0.0f);
    model.scale = glm::vec3(1.0f);
    model.rotation = glm::vec3(0.0f);
    loadedModels[modelName] = model;
    std::cout << "Model " << modelName << " loaded successfully with " << model.meshes.size() << " meshes" << std::endl;
    return true;
}

void addModelToScene(const std::string& modelName, const glm::vec3& position, const glm::vec3& scale, const glm::vec3& rotation) {
    if (loadedModels.find(modelName) == loadedModels.end()) {
        std::cerr << "Model " << modelName << " not loaded" << std::endl;
        return;
    }

    Model instance = loadedModels[modelName];

    instance.position = position;
    instance.scale = scale;
    instance.rotation = rotation;
    instance.name = modelName; 

    sceneModels.push_back(instance);
}

void createTerrain(int gridSize, float gridStep, const char* texturePath) {
  
    float size = gridSize * gridStep;
    float halfSize = size / 2.0f;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    for (int z = 0; z <= gridSize; z++) {
        for (int x = 0; x <= gridSize; x++) {
            Vertex vertex;

            vertex.position.x = x * gridStep - halfSize;
            vertex.position.y = 0.0f; 
            vertex.position.z = z * gridStep - halfSize;
            vertex.normal.x = 0.0f;
            vertex.normal.y = 1.0f;
            vertex.normal.z = 0.0f;

            vertex.texCoords.x = (float)x / gridSize;
            vertex.texCoords.y = (float)z / gridSize;
            
            vertices.push_back(vertex);
        }
    }
    
    // Indicesler - üçgenler
    for (int z = 0; z < gridSize; z++) {
        for (int x = 0; x < gridSize; x++) {
            unsigned int topLeft = z * (gridSize + 1) + x;
            unsigned int topRight = topLeft + 1;
            unsigned int bottomLeft = (z + 1) * (gridSize + 1) + x;
            unsigned int bottomRight = bottomLeft + 1;
            
            // İlk üçgen
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
            
            // İkinci üçgen
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
    
    // VAO, VBO, EBO oluştur
    glGenVertexArrays(1, &terrain.VAO);
    glGenBuffers(1, &terrain.VBO);
    glGenBuffers(1, &terrain.EBO);
    
    glBindVertexArray(terrain.VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, terrain.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
    
    // Vertex attribute pointers
    // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // Normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    // TexCoords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    
    glBindVertexArray(0);
    
    // Indice sayısını kaydet
    terrain.indexCount = indices.size();
    
    // Dokuyu yükle
    terrain.textureID = loadTexture(texturePath);
    
    // Terrain pozisyonu ve ölçeği
    terrain.position = glm::vec3(0.0f, -0.01f, 0.0f); // Biraz aşağıda, z-fighting önlemek için
    terrain.scale = 1.0f;
    
    std::cout << "Terrain created with " << vertices.size() << " vertices and " << indices.size() << " indices" << std::endl;
}

void cleanup() {
    for (auto& pair : loadedModels) {
        for (auto& mesh : pair.second.meshes) {
            glDeleteVertexArrays(1, &mesh.VAO);
            glDeleteBuffers(1, &mesh.VBO);
            glDeleteBuffers(1, &mesh.EBO);
            glDeleteTextures(1, &mesh.material.diffuseMap);
        }
    }
    glDeleteVertexArrays(1, &terrain.VAO);
    glDeleteBuffers(1, &terrain.VBO);
    glDeleteBuffers(1, &terrain.EBO);
    glDeleteTextures(1, &terrain.textureID);

    glDeleteProgram(shaderProgram);

    glDeleteFramebuffers(1, &depthMapFBO);
    glDeleteTextures(1, &depthMap);
    glDeleteProgram(shadowMapShaderProgram);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);

    WINDOW_HEIGHT = height;
    WINDOW_WIDTH = width;
}

void setupFencingSystem(const glm::vec3& barnPos) {
    float farmWidth = 90.0f;
    float farmLength = 30.0f;
    float fenceScale = 0.4f;
    float gateScale = 0.4f;
    
    float barnWidth = 10.0f;
    float barnLength = 8.0f;
    float barnOffset = -5.0f;
    
    float leftEdge = -farmWidth / 2.0f;
    float rightEdge = farmWidth / 2.0f;
    float frontEdge = farmLength / 2.0f;
    float backEdge = barnOffset - barnLength/2.0f - 2.0f;
    backEdge += 5.0f;
    

    float fenceCollisionWidth = 5.0f;  
    float fenceCollisionHeight = 2.0f; 
    float fenceCollisionDepth = 0.3f; 

    float startPosLeftBack = -5.1f;
    float fenceWidth = 4.7f;
    float xPosLeft;
    
    for (int i = 0; i < 2; i++) {
        xPosLeft = startPosLeftBack - fenceWidth * i;
        if (xPosLeft > leftEdge) {
            glm::vec3 fencePos = glm::vec3(xPosLeft, 0.0f, backEdge);

            addModelToScene("fence", fencePos, glm::vec3(fenceScale), glm::vec3(0.0f, 0.0f, 0.0f));

            collisionSystem.addCustomObject(
                AABB(fencePos - glm::vec3(fenceCollisionWidth/2, 0.0f, fenceCollisionDepth/2),
                     fencePos + glm::vec3(fenceCollisionWidth/2, fenceCollisionHeight, fenceCollisionDepth/2)),
                "BackLeftFence_" + std::to_string(i), true);
        }
    }



    addModelToScene("fence", glm::vec3(0.0f, 0.0f, backEdge), glm::vec3(0.465,0.4,0.4), glm::vec3(0.0f, 0.0f, 0.0f));
    collisionSystem.addCustomObject(
        AABB(glm::vec3(-fenceCollisionWidth/2, 0.0f, backEdge - fenceCollisionDepth/2),
             glm::vec3(fenceCollisionWidth/2, fenceCollisionHeight, backEdge + fenceCollisionDepth/2)),
        "BackMiddleFence", true);
    

    float startPosRightBack = 5.1f;
    for (int i = 0; i < 2; i++) {
        float xPos = startPosRightBack + fenceWidth * i; 
        if (xPos < rightEdge) {
            glm::vec3 fencePos = glm::vec3(xPos, 0.0f, backEdge);
      
            addModelToScene("fence", fencePos, glm::vec3(fenceScale), glm::vec3(0.0f, 0.0f, 0.0f));
            
          
            collisionSystem.addCustomObject(
                AABB(fencePos - glm::vec3(fenceCollisionWidth/2, 0.0f, fenceCollisionDepth/2),
                     fencePos + glm::vec3(fenceCollisionWidth/2, fenceCollisionHeight, fenceCollisionDepth/2)),
                "BackRightFence_" + std::to_string(i), true);
        }
    }

    float leftEndPos = xPosLeft - fenceWidth/2;
    float fenceLength = 4.7f;
    float zPosLeftForward = backEdge + fenceLength - fenceWidth * 1.5;

    for (int i = 0; i < 5; i++) {
        zPosLeftForward += fenceLength;
        glm::vec3 fencePos = glm::vec3(leftEndPos, 0.0f, zPosLeftForward);

        addModelToScene("fence", fencePos, glm::vec3(fenceScale), glm::vec3(0.0f, 90.0f, 0.0f));

        collisionSystem.addCustomObject(
            AABB(fencePos - glm::vec3(fenceCollisionDepth/2, 0.0f, fenceCollisionWidth/2),
                 fencePos + glm::vec3(fenceCollisionDepth/2, fenceCollisionHeight, fenceCollisionWidth/2)),
            "LeftSideFence_" + std::to_string(i), true);
    }


    float rightEndPos = startPosRightBack + fenceWidth * 1.5;
    float zPosRightForward = backEdge + fenceLength - fenceWidth * 1.5;

    for (int i = 0; i < 5; i++) {
        zPosRightForward += fenceLength;
        glm::vec3 fencePos = glm::vec3(rightEndPos, 0.0f, zPosRightForward);

        addModelToScene("fence", fencePos, glm::vec3(fenceScale), glm::vec3(0.0f, 90.0f, 0.0f));

        collisionSystem.addCustomObject(
            AABB(fencePos - glm::vec3(fenceCollisionDepth/2, 0.0f, fenceCollisionWidth/2),
                 fencePos + glm::vec3(fenceCollisionDepth/2, fenceCollisionHeight, fenceCollisionWidth/2)),
            "RightSideFence_" + std::to_string(i), true);
    }

    float frontZPos = std::max(zPosLeftForward, zPosRightForward) + fenceLength/2;
    float frontWidth = rightEndPos - leftEndPos;
    int numFences = 5;
    float itemWidth = frontWidth / numFences;
    int gateIndex = numFences / 2; 
    
    for (int i = 0; i < numFences; i++) {
        float xPos = leftEndPos + itemWidth * i + itemWidth/2;
        glm::vec3 itemPos = glm::vec3(xPos, 0.0f, frontZPos);

        if (i == gateIndex) {
           
            gateClosedPos = itemPos; 
            gateOpenPos = itemPos;   
            gateOpenPos.x += 2.3f;  
            gateOpenPos.z += 2.2f;   
            gateCurrentPos = gateClosedPos; 
            gateAnimationProgress = 0.0f;   

            addModelToScene("gate", gateClosedPos, glm::vec3(gateScale), glm::vec3(0.0f, 0.0f, 0.0f));

            addGateClosedCollision();
        } else {

            addModelToScene("fence", itemPos, glm::vec3(fenceScale), glm::vec3(0.0f, 0.0f, 0.0f));

            collisionSystem.addCustomObject(
                AABB(itemPos - glm::vec3(fenceCollisionWidth/2, 0.0f, fenceCollisionDepth/2),
                     itemPos + glm::vec3(fenceCollisionWidth/2, fenceCollisionHeight, fenceCollisionDepth/2)),
                "FrontFence_" + std::to_string(i), true);
            if (i == gateControlPostIndex) {
                gateControlPostPos = itemPos;
            }
            
        }
    }

}

void setupTrees(GLuint blendedShader, GLuint barkTexture, GLuint leavesTexture, GLuint windmillTexture) {
    float farmWidth = 90.0f;
    float farmLength = 30.0f;
    float leftEdge = -farmWidth / 2.0f;
    float rightEdge = farmWidth / 2.0f;
    float frontEdge = farmLength / 2.0f;
    float backEdge = -12.0f;

    float baseTreeRadius = 10.0f;  
    float baseTreeHeight = 300.0f;  
    float baseWindmillRadius = 2.2f; 
    float baseWindmillHeight = 400.0f; 

    glm::vec3 tree1Pos = glm::vec3(12.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + 2.0f+ (frontEdge - backEdge) * 0.3f);  
    glm::vec3 tree1Scale = glm::vec3(0.03f, 0.04f, 0.03f);
    
    Model tree1 = loadedModels["tree"];
    tree1.position = tree1Pos;
    tree1.scale = tree1Scale;
    tree1.rotation = glm::vec3(0.0f, 45.0f, 0.0f);
    tree1.name = "normalTree";
    
    for (size_t i = 0; i < tree1.meshes.size(); i++) {
        if (i == 0) {
            tree1.meshes[i].material.diffuseMap = leavesTexture;
            tree1.meshes[i].material.hasTexture = true;
        } 
        else if (i == 1 || i == 3) {
            tree1.meshes[i].material.diffuseMap = barkTexture;
            tree1.meshes[i].material.hasTexture = true;
        }
        else if (i == 2 || i == 4) {
            tree1.meshes[i].material.diffuseMap = barkTexture;
            tree1.meshes[i].material.hasTexture = true;
        }
    }
    sceneModels.push_back(tree1);
    float tree1CollisionRadius = baseTreeRadius * tree1Scale.x;
    float tree1CollisionHeight = baseTreeHeight * tree1Scale.y;
    
    AABB tree1AABB(tree1Pos - glm::vec3(tree1CollisionRadius, 0.0f, tree1CollisionRadius),
                   tree1Pos + glm::vec3(tree1CollisionRadius, tree1CollisionHeight, tree1CollisionRadius));
    
    collisionSystem.addCustomObject(tree1AABB, "Tree1_Normal", true);
    glm::vec3 tree2Pos = glm::vec3(16.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + 4.0f+ (frontEdge - backEdge) * 0.3f);
    
    glm::vec3 tree2Scale = glm::vec3(0.03f, 0.06f, 0.03f);
    
    Model tree2 = loadedModels["tree"];
    tree2.position = tree2Pos;
    tree2.scale = tree2Scale;
    tree2.rotation = glm::vec3(0.0f, -30.0f, 0.0f);
    tree2.name = "blendedTree";
    tree2.useBlendedTextures = true;
    tree2.blendedShaderID = blendedShader;
    tree2.customTextureID = barkTexture;
    tree2.secondTextureID = leavesTexture;
    tree2.blendFactor = 0.6f;
    tree2.isRotating = true; 

    for (auto& mesh : tree2.meshes) {
        mesh.material.hasTexture = false;
        mesh.material.diffuse = glm::vec3(1.0f);
    }
    sceneModels.push_back(tree2);
    float tree2CollisionRadius = baseTreeRadius * tree2Scale.x;
    float tree2CollisionHeight = baseTreeHeight * tree2Scale.y;
    
    AABB tree2AABB(tree2Pos - glm::vec3(tree2CollisionRadius, 0.0f, tree2CollisionRadius),
                   tree2Pos + glm::vec3(tree2CollisionRadius, tree2CollisionHeight, tree2CollisionRadius));
    
    collisionSystem.addCustomObject(tree2AABB, "Tree2_Blended", true);

    glm::vec3 tree3Pos = glm::vec3(32.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + (frontEdge - backEdge) * 0.3f);

    glm::vec3 tree3Scale = glm::vec3(0.03f, 0.05f, 0.03f);
    
    Model tree3 = loadedModels["tree"];
    tree3.position = tree3Pos;
    tree3.scale = tree3Scale;
    tree3.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    tree3.name = "rotatingTree";
    tree3.isRotating = false;
    
    for (size_t i = 0; i < tree3.meshes.size(); i++) {
        if (i == 0) {
            tree3.meshes[i].material.diffuseMap = leavesTexture;
            tree3.meshes[i].material.hasTexture = true;
        } 
        else if (i == 1 || i == 3) {
            tree3.meshes[i].material.diffuseMap = barkTexture;
            tree3.meshes[i].material.hasTexture = true;
        }
        else if (i == 2 || i == 4) {
            tree3.meshes[i].material.diffuseMap = barkTexture;
            tree3.meshes[i].material.hasTexture = true;
        }
    }
    sceneModels.push_back(tree3);

    float tree3CollisionRadius = baseTreeRadius * tree3Scale.x;
    float tree3CollisionHeight = baseTreeHeight * tree3Scale.y;
    
    AABB tree3AABB(tree3Pos - glm::vec3(tree3CollisionRadius, 0.0f, tree3CollisionRadius),
                   tree3Pos + glm::vec3(tree3CollisionRadius, tree3CollisionHeight, tree3CollisionRadius));
    
    collisionSystem.addCustomObject(tree3AABB, "Tree3_Big", true);

    glm::vec3 windmillPos = glm::vec3(32.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + 6.0f + (frontEdge - backEdge) * 0.3f) + glm::vec3(0.0f, 0.0f, 11.0f);
    
    glm::vec3 windmillScale = glm::vec3(0.5f, 0.5f, 0.5f);
    
    Model windmill = loadedModels["windmill"];
    windmill.position = windmillPos;
    windmill.scale = windmillScale;
    windmill.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    windmill.name = "windmill";
    windmill.isRotating = true; 
    
    sceneModels.push_back(windmill);

    float windmillCollisionRadius = baseWindmillRadius * windmillScale.x;
    float windmillCollisionHeight = baseWindmillHeight * windmillScale.y;
    
    AABB windmillAABB(windmillPos - glm::vec3(windmillCollisionRadius, 0.0f, windmillCollisionRadius),
                      windmillPos + glm::vec3(windmillCollisionRadius, windmillCollisionHeight, windmillCollisionRadius));
    
    collisionSystem.addCustomObject(windmillAABB, "Windmill", true);
}

void setupBarrel(GLuint blendedShader, GLuint barrel1Tex, GLuint barrel2Tex){
    float farmWidth = 90.0f;
    float farmLength = 30.0f;
    float leftEdge = -farmWidth / 2.0f;
    float rightEdge = farmWidth / 2.0f;
    float frontEdge = farmLength / 2.0f;
    float backEdge = -12.0f;
 
    float baseCollisionRadius = 50.0f; 
    float baseCollisionHeight = 200.0f;  
    
    glm::vec3 barrel1Pos = glm::vec3(12.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + 14.0f+ (frontEdge - backEdge) * 0.3f);
    
    glm::vec3 barrel1Scale = glm::vec3(0.01f, 0.01f, 0.01f);
    
    Model barrel1 = loadedModels["barrel"];
    barrel1.position = barrel1Pos;
    barrel1.scale = barrel1Scale;
    barrel1.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    barrel1.name = "barrel1";
    
    for (size_t i = 0; i < barrel1.meshes.size(); i++) {
        barrel1.meshes[i].material.diffuseMap = barrel1Tex;
        barrel1.meshes[i].material.hasTexture = true;
    }
    sceneModels.push_back(barrel1);

    float barrel1CollisionRadius = baseCollisionRadius * barrel1Scale.x;
    float barrel1CollisionHeight = baseCollisionHeight * barrel1Scale.y;
    
    AABB barrel1AABB(barrel1Pos - glm::vec3(barrel1CollisionRadius, 0.0f, barrel1CollisionRadius),
                     barrel1Pos + glm::vec3(barrel1CollisionRadius, barrel1CollisionHeight, barrel1CollisionRadius));
    
    collisionSystem.addCustomObject(barrel1AABB, "Barrel1", true);

    glm::vec3 barrel2Pos = glm::vec3(12.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + 17.0f+ (frontEdge - backEdge) * 0.3f);
    
    glm::vec3 barrel2Scale = glm::vec3(0.015f, 0.015f, 0.015f);
    
    Model barrel2 = loadedModels["barrel"];
    barrel2.position = barrel2Pos;
    barrel2.scale = barrel2Scale;
    barrel2.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    barrel2.name = "blendedBarrel";
    barrel2.useBlendedTextures = true;
    barrel2.blendedShaderID = blendedShader;
    barrel2.customTextureID = barrel1Tex;
    barrel2.secondTextureID = barrel2Tex;
    barrel2.blendFactor = 0.6f;

    for (auto& mesh : barrel2.meshes) {
        mesh.material.hasTexture = false;
        mesh.material.diffuse = glm::vec3(1.0f);
    }
    sceneModels.push_back(barrel2);

    float barrel2CollisionRadius = baseCollisionRadius * barrel2Scale.x;
    float barrel2CollisionHeight = baseCollisionHeight * barrel2Scale.y;
    
    AABB barrel2AABB(barrel2Pos - glm::vec3(barrel2CollisionRadius, 0.0f, barrel2CollisionRadius),
                     barrel2Pos + glm::vec3(barrel2CollisionRadius, barrel2CollisionHeight, barrel2CollisionRadius));
    
    collisionSystem.addCustomObject(barrel2AABB, "Barrel2_Blended", true);

    glm::vec3 barrel3Pos = glm::vec3(12.0f + leftEdge + (rightEdge - leftEdge) * 0.25f, 0.0f, 
                                   backEdge + 20.0f+ (frontEdge - backEdge) * 0.3f);

    glm::vec3 barrel3Scale = glm::vec3(0.017f, 0.017f, 0.017f);
    
    Model barrel3 = loadedModels["barrel"];
    barrel3.position = barrel3Pos;
    barrel3.scale = barrel3Scale;
    barrel3.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    barrel3.name = "barrel3";
    barrel3.isRotating = true;
    
    for (size_t i = 0; i < barrel3.meshes.size(); i++) {
        barrel3.meshes[i].material.diffuseMap = barrel1Tex;
        barrel3.meshes[i].material.hasTexture = true;
    }
    sceneModels.push_back(barrel3);

    float barrel3CollisionRadius = baseCollisionRadius * barrel3Scale.x;
    float barrel3CollisionHeight = baseCollisionHeight * barrel3Scale.y;
    
    AABB barrel3AABB(barrel3Pos - glm::vec3(barrel3CollisionRadius, 0.0f, barrel3CollisionRadius),
                     barrel3Pos + glm::vec3(barrel3CollisionRadius, barrel3CollisionHeight, barrel3CollisionRadius));
    
    collisionSystem.addCustomObject(barrel3AABB, "Barrel3_Rotating", true);
    

}

void setupFarmScene() {
    loadModel("textures/OpenBarn.obj", "barn");
    loadModel("textures/Fence.obj", "fence");
    loadModel("textures/Gate.obj", "gate");
    loadModel("textures/Tree.obj", "tree");
    loadModel("textures/CircleBulb.obj", "lightbulb");
    loadModel("textures/TowerWindmill.obj", "windmill");
    loadModel("textures/barrel.obj", "barrel");

    GLuint barkTexture = loadTexture("textures/tree_bark.jpg");
    GLuint leavesTexture = loadTexture("textures/tree_leaves.jpg");
    GLuint streetLampTexture = loadTexture("textures/CircleBulb.png");
    GLuint streetLampTextureGreen = loadTexture("textures/CircleBulbGreen.png");
    GLuint streetLampTextureWhite = loadTexture("textures/CircleBulbWhite.png");

    GLuint blendedShader = createShader(vertexShaderSource, blendedTreeFragmentShaderSource);
    GLuint lampShader = createShader(lampVertexShaderSource, lampFragmentShaderSource);
  
    collisionSystem.clear();

    float baseLampRadius = 1.0f;  
    float baseLampHeight = 200.0f; 

    glm::vec3 barnPos = glm::vec3(0.0f, 0.0f, -5.0f);
    glm::vec3 mainLampPos = barnPos + glm::vec3(10.0f, 0.0f, 3.0f);
    glm::vec3 mainLampLightPos = barnPos + glm::vec3(10.0f, 4.0f, 3.0f);
    
    glm::vec3 mainLampScale = glm::vec3(0.2f);
    
    Model mainLamp = loadedModels["lightbulb"];
    mainLamp.position = mainLampPos;
    mainLamp.scale = mainLampScale;
    mainLamp.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    mainLamp.name = "mainStreetLamp";
    mainLamp.useCustomShader = true;
    mainLamp.customShaderID = lampShader;
    mainLamp.useCustomTexture = true;
    mainLamp.customTextureID = streetLampTexture;
    sceneModels.push_back(mainLamp);

    float mainLampCollisionRadius = baseLampRadius * mainLampScale.x;
    float mainLampCollisionHeight = baseLampHeight * mainLampScale.y;
    
    AABB mainLampAABB(mainLampPos - glm::vec3(mainLampCollisionRadius, 0.0f, mainLampCollisionRadius),
                      mainLampPos + glm::vec3(mainLampCollisionRadius, mainLampCollisionHeight, mainLampCollisionRadius));
    
    collisionSystem.addCustomObject(mainLampAABB, "MainLamp", true);

    PointLight mainLampLight(mainLampLightPos, glm::vec3(1.0f, 0.6f, 0.2f), 10.0f);
    mainLampLight.constant = 1.0f;
    mainLampLight.linear = 0.14f;
    mainLampLight.quadratic = 0.6f;
    lightingSystem.addPointLight(mainLampLight);

    glm::vec3 secondLampPos = barnPos + glm::vec3(-8.0f, 0.0f, 18.25f);
    glm::vec3 secondLampLightPos = barnPos + glm::vec3(-8.0f, 4.0f, 18.25f);
    
    glm::vec3 secondLampScale = glm::vec3(0.25f);
    
    Model secondLamp = loadedModels["lightbulb"];
    secondLamp.position = secondLampPos;
    secondLamp.scale = secondLampScale;
    secondLamp.rotation = glm::vec3(0.0f, 45.0f, 0.0f);
    secondLamp.name = "secondStreetLamp";
    secondLamp.useCustomShader = true;
    secondLamp.customShaderID = lampShader;
    secondLamp.useCustomTexture = true;
    secondLamp.customTextureID = streetLampTextureWhite;
    sceneModels.push_back(secondLamp);

    float secondLampCollisionRadius = baseLampRadius * secondLampScale.x;
    float secondLampCollisionHeight = baseLampHeight * secondLampScale.y;
    
    AABB secondLampAABB(secondLampPos - glm::vec3(secondLampCollisionRadius, 0.0f, secondLampCollisionRadius),
                        secondLampPos + glm::vec3(secondLampCollisionRadius, secondLampCollisionHeight, secondLampCollisionRadius));
    
    collisionSystem.addCustomObject(secondLampAABB, "SecondLamp", true);

    PointLight secondLampLight(secondLampLightPos, glm::vec3(0.8f, 0.9f, 1.0f), 5.0f);
    secondLampLight.constant = 1.0f;
    secondLampLight.linear = 0.35f;
    secondLampLight.quadratic = 0.044f;
    lightingSystem.addPointLight(secondLampLight);

    glm::vec3 thirdLampPos = barnPos + glm::vec3(8.0f, 0.0f, 20.0f);
    glm::vec3 thirdLampLightPos = barnPos + glm::vec3(8.0f, 4.0f, 20.0f);
    
    glm::vec3 thirdLampScale = glm::vec3(0.15f);
    
    Model thirdLamp = loadedModels["lightbulb"];
    thirdLamp.position = thirdLampPos;
    thirdLamp.scale = thirdLampScale;
    thirdLamp.rotation = glm::vec3(0.0f, -45.0f, 0.0f);
    thirdLamp.name = "thirdStreetLamp";
    thirdLamp.useCustomShader = true;
    thirdLamp.customShaderID = lampShader;
    thirdLamp.useCustomTexture = true;
    thirdLamp.customTextureID = streetLampTextureGreen;
    sceneModels.push_back(thirdLamp);

    float thirdLampCollisionRadius = baseLampRadius * thirdLampScale.x;
    float thirdLampCollisionHeight = baseLampHeight * thirdLampScale.y;
    
    AABB thirdLampAABB(thirdLampPos - glm::vec3(thirdLampCollisionRadius, 0.0f, thirdLampCollisionRadius),
                       thirdLampPos + glm::vec3(thirdLampCollisionRadius, thirdLampCollisionHeight, thirdLampCollisionRadius));
    
    collisionSystem.addCustomObject(thirdLampAABB, "ThirdLamp", true);

    PointLight thirdLampLight(thirdLampLightPos, glm::vec3(0.6f, 1.0f, 0.7f), 5.0f);
    thirdLampLight.constant = 1.0f;
    thirdLampLight.linear = 0.35f;
    thirdLampLight.quadratic = 0.44f;
    lightingSystem.addPointLight(thirdLampLight);

    glm::vec3 _LampPos = barnPos + glm::vec3(-10.0f, 0.0f, 1.0f);
    glm::vec3 _LampLightPos = barnPos + glm::vec3(-10.0f, 4.0f, 1.0f);
    
    glm::vec3 _LampScale = glm::vec3(0.2f);
    
    Model _Lamp = loadedModels["lightbulb"];
    _Lamp.position = _LampPos;
    _Lamp.scale = _LampScale;
    _Lamp.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    _Lamp.name = "fourthStreetLamp";
    _Lamp.useCustomShader = true;
    _Lamp.customShaderID = lampShader;
    _Lamp.useCustomTexture = true;
    _Lamp.customTextureID = streetLampTexture;
    sceneModels.push_back(_Lamp);

    float _LampCollisionRadius = baseLampRadius * _LampScale.x;
    float _LampCollisionHeight = baseLampHeight * _LampScale.y;
    
    AABB _LampAABB(_LampLightPos - glm::vec3(_LampCollisionRadius, 0.0f, _LampCollisionRadius),
                      _LampLightPos + glm::vec3(_LampCollisionRadius, _LampCollisionHeight, _LampCollisionRadius));
    
    collisionSystem.addCustomObject(_LampAABB, "FourthLamp", true);

    PointLight _LampLight(_LampLightPos, glm::vec3(1.0f, 0.6f, 0.2f), 3.0f);
    _LampLight.constant = 1.0f;
    _LampLight.linear = 0.09f;
    _LampLight.quadratic = 0.032f;
    lightingSystem.addPointLight(_LampLight);

    setupFencingSystem(barnPos); 
    setupTrees(blendedShader, barkTexture, leavesTexture, streetLampTextureWhite); 
    setupBarrel(blendedShader, barkTexture, leavesTexture);

    lightingSystem.setDirectionalLight(
        glm::vec3(0.0f, -1.0f, -0.7f),
        glm::vec3(0.7f, 0.8f, 1.0f),
        1.5f
    );

    createTerrain(60, 1.0f, "textures/floor.jpg");
}

int main() {

   if (!glfwInit()) {
       std::cerr << "Failed to initialize GLFW" << std::endl;
       return -1;
   }
   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

   GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Farm Scene", NULL, NULL);
   if (!window) {
       std::cerr << "Failed to create GLFW window" << std::endl;
       glfwTerminate();
       return -1;
   }
   glfwMakeContextCurrent(window);
   glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
   glfwSetCursorPosCallback(window, mouse_callback);
   glfwSetKeyCallback(window, key_callback);
   glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

   if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
       std::cerr << "Failed to initialize GLAD" << std::endl;
       return -1;
   }

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   shaderProgram = createShader(vertexShaderSource, fragmentShaderSource);
   GLuint terrainShader = createShader(terrainVertexShaderSource, terrainFragmentShaderSource);
   shadowMapShaderProgram = createShader(shadowMapVertexShaderSource, shadowMapFragmentShaderSource);
   
   if (!shaderProgram || !terrainShader) { 
       std::cerr << "Failed to create shader programs" << std::endl;
       return -1;
   }

   setupShadowMap();

   setupFarmScene();

   cameraPos = playerPosition;

   setupInputCallbacks(window);

   while (!glfwWindowShouldClose(window)) {

    frameCount++;
    float currentTime = glfwGetTime();
    if (currentTime - lastTime >= 1.0f) {
        float fps = frameCount / (currentTime - lastTime);
        std::cout << "FPS: " << (int)fps << std::endl;
        frameCount = 0;
        lastTime = currentTime;
    }


       float currentFrame = glfwGetTime();
       deltaTime = currentFrame - lastFrame;
       lastFrame = currentFrame;
 
       processInput(window);
        updateGateAnimation(deltaTime);

       // SHADOW MAP PASS
       float near_plane = 1.0f, far_plane = 50.0f;
       float orthoSize = 30.0f;
       glm::vec3 lightDir = lightingSystem.getDirectionalLightDir();
       glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, near_plane, far_plane);
       glm::mat4 lightView = glm::lookAt(-lightDir * 20.0f, 
                                        glm::vec3(0.0f),   
                                        glm::vec3(0.0f, 1.0f, 0.0f)); 
       glm::mat4 lightSpaceMatrix = lightProjection * lightView;

       glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
       glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
       glClear(GL_DEPTH_BUFFER_BIT);

       glUseProgram(shadowMapShaderProgram);
       glUniformMatrix4fv(glGetUniformLocation(shadowMapShaderProgram, "lightSpaceMatrix"), 
                          1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
 
       glm::mat4 terrainModelShadow = glm::mat4(1.0f);
       terrainModelShadow = glm::translate(terrainModelShadow, terrain.position);
       terrainModelShadow = glm::scale(terrainModelShadow, glm::vec3(terrain.scale));
       
       glUniformMatrix4fv(glGetUniformLocation(shadowMapShaderProgram, "model"), 
                          1, GL_FALSE, glm::value_ptr(terrainModelShadow));
       
       glBindVertexArray(terrain.VAO);
       glDrawElements(GL_TRIANGLES, terrain.indexCount, GL_UNSIGNED_INT, 0);
       
       for (const auto& model : sceneModels) {
           glm::mat4 modelMatrix = glm::mat4(1.0f);
            if (model.isRotating) {
                float currentTime = glfwGetTime();
                modelMatrix = calculateRotatingTreeMatrix(currentTime, model.position, model.scale);
            } 
            
            else {
                modelMatrix = glm::translate(modelMatrix, model.position);
                modelMatrix = glm::rotate(modelMatrix, glm::radians(model.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
                modelMatrix = glm::rotate(modelMatrix, glm::radians(model.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
                modelMatrix = glm::rotate(modelMatrix, glm::radians(model.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
                modelMatrix = glm::scale(modelMatrix, model.scale);
            }
            
            glUniformMatrix4fv(glGetUniformLocation(shadowMapShaderProgram, "model"), 
                            1, GL_FALSE, glm::value_ptr(modelMatrix));
            
            for (const auto& mesh : model.meshes) {
                glBindVertexArray(mesh.VAO);
                glDrawElements(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0);
            }
       }
       
       glBindFramebuffer(GL_FRAMEBUFFER, 0);

       // NORMAL RENDERING PASS
       int w,h;
       glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
       glClearColor(0.02f, 0.02f, 0.08f, 1.0f);
       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = glm::perspective(glm::radians(fov), 
        (float)w / (float)h, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
  
       glUseProgram(terrainShader);

       glUniformMatrix4fv(glGetUniformLocation(terrainShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
       glUniformMatrix4fv(glGetUniformLocation(terrainShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
       glUniformMatrix4fv(glGetUniformLocation(terrainShader, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

       glm::mat4 terrainModel = glm::mat4(1.0f);
       terrainModel = glm::translate(terrainModel, terrain.position);
       terrainModel = glm::scale(terrainModel, glm::vec3(terrain.scale));
       
       glUniformMatrix4fv(glGetUniformLocation(terrainShader, "model"), 1, GL_FALSE, glm::value_ptr(terrainModel));
       lightingSystem.sendToShader(terrainShader, cameraPos);

       glActiveTexture(GL_TEXTURE0);
       glBindTexture(GL_TEXTURE_2D, terrain.textureID);
       glUniform1i(glGetUniformLocation(terrainShader, "terrainTexture"), 0);
       
       // Shadow map
       glActiveTexture(GL_TEXTURE1);
       glBindTexture(GL_TEXTURE_2D, depthMap);
       glUniform1i(glGetUniformLocation(terrainShader, "shadowMap"), 1);

       // Terrain çizme
       glBindVertexArray(terrain.VAO);
       glDrawElements(GL_TRIANGLES, terrain.indexCount, GL_UNSIGNED_INT, 0);
       glBindVertexArray(0);

       // Sonra modeller
       glUseProgram(shaderProgram);

       lightingSystem.sendToShader(shaderProgram, cameraPos);

       glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
       glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
       glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

       glActiveTexture(GL_TEXTURE1);
       glBindTexture(GL_TEXTURE_2D, depthMap);
       glUniform1i(glGetUniformLocation(shaderProgram, "shadowMap"), 1);

    for (const auto& model : sceneModels) {

        glm::mat4 modelMatrix = glm::mat4(1.0f);

        if (model.isRotating) {

            float currentTime = glfwGetTime(); 
            modelMatrix = calculateRotatingTreeMatrix(currentTime, model.position, model.scale);
        } else {

            modelMatrix = glm::translate(modelMatrix, model.position);
            modelMatrix = glm::rotate(modelMatrix, glm::radians(model.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
            modelMatrix = glm::rotate(modelMatrix, glm::radians(model.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
            modelMatrix = glm::rotate(modelMatrix, glm::radians(model.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
            modelMatrix = glm::scale(modelMatrix, model.scale);
        }
 
        GLuint currentShader = shaderProgram; 
    
        if (model.useBlendedTextures && model.blendedShaderID != 0) {
               currentShader = model.blendedShaderID;
               glUseProgram(currentShader);
               
               glUniformMatrix4fv(glGetUniformLocation(currentShader, "model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
               glUniformMatrix4fv(glGetUniformLocation(currentShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
               glUniformMatrix4fv(glGetUniformLocation(currentShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
               glUniformMatrix4fv(glGetUniformLocation(currentShader, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

                   lightingSystem.sendToShader(currentShader, cameraPos);

               glUniform1f(glGetUniformLocation(currentShader, "blendFactor"), model.blendFactor);
               
               glActiveTexture(GL_TEXTURE0);
               glBindTexture(GL_TEXTURE_2D, model.customTextureID); 
               glUniform1i(glGetUniformLocation(currentShader, "texture1"), 0);
               
               glActiveTexture(GL_TEXTURE1);
               glBindTexture(GL_TEXTURE_2D, model.secondTextureID); 
               glUniform1i(glGetUniformLocation(currentShader, "texture2"), 1);

                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, depthMap);
                glUniform1i(glGetUniformLocation(currentShader, "shadowMap"), 2);
               
               glUniform3fv(glGetUniformLocation(currentShader, "material_diffuse"), 1, glm::value_ptr(glm::vec3(1.0f)));
               glUniform3fv(glGetUniformLocation(currentShader, "material_specular"), 1, glm::value_ptr(glm::vec3(0.5f)));
               glUniform3fv(glGetUniformLocation(currentShader, "material_ambient"), 1, glm::value_ptr(glm::vec3(0.3f)));
               glUniform1f(glGetUniformLocation(currentShader, "material_shininess"), 32.0f);
               
           } 
           else 
    if (model.useCustomShader && model.customShaderID != 0) {
        currentShader = model.customShaderID;
        glUseProgram(currentShader);
        
        glUniformMatrix4fv(glGetUniformLocation(currentShader, "model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glUniformMatrix4fv(glGetUniformLocation(currentShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(currentShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(currentShader, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
        
        lightingSystem.sendToShader(currentShader, cameraPos);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, model.customTextureID);
        glUniform1i(glGetUniformLocation(currentShader, "glassTexture"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        glUniform1i(glGetUniformLocation(currentShader, "shadowMap"), 1);
        
    }
           else {
                currentShader = shaderProgram;
                glUseProgram(currentShader);
                glUniformMatrix4fv(glGetUniformLocation(currentShader, "model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
                glUniformMatrix4fv(glGetUniformLocation(currentShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
                glUniformMatrix4fv(glGetUniformLocation(currentShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
                glUniformMatrix4fv(glGetUniformLocation(currentShader, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
                
                lightingSystem.sendToShader(currentShader, cameraPos);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, depthMap);
                glUniform1i(glGetUniformLocation(currentShader, "shadowMap"), 1);
           }

        for (const auto& mesh : model.meshes) {

            if (model.useBlendedTextures) {
                   
               } else {

            glUniform3fv(glGetUniformLocation(currentShader, "material_diffuse"), 1, glm::value_ptr(mesh.material.diffuse));
        glUniform3fv(glGetUniformLocation(currentShader, "material_specular"), 1, glm::value_ptr(mesh.material.specular));
        glUniform3fv(glGetUniformLocation(currentShader, "material_ambient"), 1, glm::value_ptr(mesh.material.ambient));
        glUniform1f(glGetUniformLocation(currentShader, "material_shininess"), mesh.material.shininess);
        
        glUniform3fv(glGetUniformLocation(currentShader, "material_emission"), 1, 
                    glm::value_ptr(mesh.material.hasEmission ? mesh.material.emission : glm::vec3(0.0f)));
        glUniform1i(glGetUniformLocation(currentShader, "has_emission"), mesh.material.hasEmission);

        if (model.useCustomTexture) {
            glUniform1i(glGetUniformLocation(currentShader, "has_texture"), true);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, model.customTextureID);
        } else {
            glUniform1i(glGetUniformLocation(currentShader, "has_texture"), mesh.material.hasTexture);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.material.diffuseMap);
        }
        
        glUniform1i(glGetUniformLocation(currentShader, "diffuseMap"), 0);
               }
            glBindVertexArray(mesh.VAO);
            glDrawElements(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

    }

        std::string modeStr = (currentCameraMode == PLAYER_MODE) ? "Player Mode" : "God Mode";
            std::string title = "Farm Scene - " + modeStr + " - Position: (" + 
                            std::to_string(cameraPos.x) + ", " + 
                            std::to_string(cameraPos.y) + ", " + 
                            std::to_string(cameraPos.z) + ") - Press N to switch mode";
            glfwSetWindowTitle(window, title.c_str());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    cleanup();
    glfwTerminate(); 
    return 0;
    }



    
