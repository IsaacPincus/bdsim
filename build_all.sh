source ../.venv/bin/activate

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBDSIM_PYTHON=ON \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
cmake --build build -j
cp build/_bdsim*.so python/bdsim/        # place the extension in the package
pip install -e .
make -C docs html
