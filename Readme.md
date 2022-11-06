# Yolov7 with ByteTrack

1. Clone repo.

```
git clone https://github.com/axcelerateai/yolov7-bytetrack-streamlit.git
cd yolov7-bytetrack-streamlit
```

2. Install requirements.

### Pip 

```
python3 -m venv .env
source .env/bin/activate
```
```
pip install Cython numpy
```
```
pip install -r requirements.txt
```

- [Note]: `cython_bbox` have no windows distribution on pypi. If you're a windows user then run following command to install `cython_bbox` from source.

```
# for windows
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

# for linux
pip install cython-bbox

```

### conda

```
conda env create -f environment.yml
```

```
conda activate yolov7_bytetrack
```

- [Note]: `cython_bbox` have no windows distribution on pypi. If you're a windows user then run following command to install `cython_bbox` from source.

```
# for windows
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

# for linux
pip install cython-bbox

```


3. Download weights.

```
python download_weights.py
```

4. Run stremlit server

```
streamlit run yolov7-tiny-demo.py --server.port [LPORT]
```
- `LPORT` = Local port of system

### Test yolov7-tiny

- To run Yolov7-Tiny 
```
streamlit run yolov7-tiny-demo.py --server.port 2085
```

### Test yolov7
```
streamlit run yolov7-demo.py --server.port 2085
```
### Test yolor
```
streamlit run yolor-demo.py --server.port 2085
```
