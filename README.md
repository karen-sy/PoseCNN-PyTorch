## Setup for deep learning baseline with real-world data

This component of Jax3DP3 runs end-to-end PoseCNN (segmentation) and DenseFusion (pose estimation) to detect and localize objects given a single RGB-D image. (Note: The model-training capability of this module has not been tested, and the scripts assume the existence of pretrained .pth models for PoseCNN and DenseFusion.)


### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above
- Python3

### Installation ([Reference](https://github.com/yuxng/PoseCNN/issues/73))

1. Install [PyTorch](https://pytorch.org/)

2. Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)

    ```Shell
        # change eigen version
        apt remove libeigen3-dev
        cd /opt
        git clone https://gitlab.com/libeigen/eigen.git
        cd eigen
        git checkout 3.3.0
        mkdir build
        cd build
        cmake ..
        make
        make install
        # make a symbolic link
        cd /usr/local/include
        ln -sf eigen3/Eigen Eigen
        ln -sf eigen3/unsupported unsupported
        ln -sf eigen3/Eigen /usr/include/Eigen
        # TODO make a patch in
        #/usr/local/include/eigen3/Core
        #change: include <math_functions.hpp> TO: include <cuda_runtime.h>
        cp -r /usr/local/include/eigen3 /usr/include/eigen3
    ```

3. Install Sophus from the Github source code [here](https://github.com/yuxng/Sophus)
    
    ```Shell
        cd /opt
        git clone https://github.com/strasdat/Sophus
        cd Sophus
        git reset --hard ceb6380a1584b300e687feeeea8799353d48859f
        # TODO include patch in CMake  (do we still need?)
        #-find_package(Eigen3 REQUIRED)
        #+find_package(PkgConfig)
        #+pkg_search_module(Eigen3 REQUIRED eigen3)
        # add -Wno-error=deprecated-copy
        mkdir build
        cd build
        cmake ..
        make
        make install
    ```

4. Install python packages
   ```Shell
   pip install -r requirement.txt
   ```

5. Initialize the submodules in ycb_render
   ```Shell
   git submodule update --init --recursive
   ```

6. Compile the new layers under $ROOT/lib/layers (for PoseCNN)
    ```Shell
    cd $ROOT/lib/layers
    python setup.py install
    ```

7. Compile cython components (for PoseCNN)
    ```Shell
    cd $ROOT/lib/utils
    python setup.py build_ext --inplace
    ```

8. Compile the ycb_render in $ROOT/ycb_render (for PoseCNN)
    ```Shell
    cd $ROOT/ycb_render
    python setup.py develop
    ```

### Download (Also see: [PoseCNN](https://github.com/NVlabs/PoseCNN-PyTorch#download), [DenseFusion](https://github.com/j96w/DenseFusion#datasets))

- 3D models of YCB Objects [here](https://drive.google.com/file/d/1PTNmhd-eSq0fwSPv0nvQN8h_scR1v-UJ/view?usp=sharing) (3G). Save as $ROOT/data/models or use a symbol link.

- Pre-trained PoseCNN checkpoints [here](https://drive.google.com/file/d/1-ECAkkTRfa1jJ9YBTzf04wxCGw6-m5d4/view?usp=sharing) (4G). Save files as $ROOT/data/trained_checkpoints/posecnn or use a symbol link.

- Pre-trained DenseFusion checkpoints [here](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7). Save trained_checkpoints.zip into $ROOT/trained_checkpoints/densefusion or use a symbol link.


### PoseCNN Demo

- First test the PoseCNN model with the provided demo.

run the following script
    ```Shell
    ./experiments/scripts/posecnn/demo.sh
    ```

- Results will be in $ROOT/datasets/posecnn/demo.

### PoseCNN + DenseFusion Demo

- Now test the full object detection and pose estimation workflow.

run the following python script
    ```Shell
    python tools/test_image_pandas_ycb_full.py   # TODO parse args
    ```
- Results will be in experiments/eval_result/pandas.  # TODO

### Testing on custom data containing YCB objects
- This is to run the posecnn+densefusion test with your own data
- For each hxw image frame you wish to test, create a .pik file containing the fields:
    - `rgb`: (h,w,3) rgb data
    - `depth`: (h,w) depth data
    - `factor_depth`: a (divisive, not multiplicative) scaling factor for the depth data
    - `intrinsics` : a 3x3 camera intrinsics matrix
- Place the folder containing these .pik files as a subfolder within $ROOT/datasets/pandas/data.