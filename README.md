# Vertex model of amnioserosa tissue in Drosophila dorsal closure

Vertex Model of Amnioserosa Tissue in Drosophila Dorsal Closure
This repository contains the code used in the paper ["Minimal vertex model explains how the amnioserosa avoids fluidization during Drosophila dorsal closure"](https://www.biorxiv.org/content/10.1101/2023.12.20.572544v2).

The code implements a 2D Vertex model to explore the origin of rigidity in amnioserosa tissue during dorsal closure.

## Prerequisites

Ensure all prerequisite libraries listed in the [cellGPU documentation](https://dmsussman.gitlab.io/cellGPUdocumentation/) are properly installed and included in your path.

For Ubuntu users, we recommend following the installation guide provided [here](https://github.com/deppinto/cellGPU_install).

## Project Structure

The main components of this project are:

1. `inc/`: Header files
2. `obj/`: Object files
3. `src/`: Source code files

## Compilation and Execution

To compile and run the code:

1. Clone this repository
2. Navigate to the project directory
3. Ensure all required libraries are installed and paths in the makefile are correct
4. Run the makefile:
   ```
   make
   ```
5. This will create an executable named `main.out`
6. Run the simulation with appropriate arguments:
   ```
   ./main.out [arguments]
   ``` 
   For a list of available arguments and their descriptions, please refer to the `main.cpp` file.
   
   
## Acknowledgements

This project uses the cellGPU package. For more information, visit the [official cellGPU documentation](https://dmsussman.gitlab.io/cellGPUdocumentation/).



