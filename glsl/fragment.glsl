#version 330 core

// Interpolated values from the vertex shaders
// in vec3 fragmentColor;

// Ouput data
out vec4 FragColor;

void main(){

	// Output color = color specified in the vertex shader,
	// interpolated between all 3 surrounding vertices
	FragColor = fragmentColor;

}