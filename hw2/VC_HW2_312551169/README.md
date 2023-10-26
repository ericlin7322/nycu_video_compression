# HW 2

## Environment Setup

```pip install -r requirements.txt```

## Running

I using the multiprocessing to speed up the process, if you don't wannt to use multiprocessing, you can run

```python main.py```

it will use the default -j1

if you want to speed up the process, you can type this with full CPU thread

```python main.py -j$(nproc)```

or you can choose many CPU thread you want to use in your computer by ```-j{num}```, for example:

```python main.py -j4```

## Result

origin:  

![](lena.png)

gray:  

![](lena_gray.png)

## 2D-DCT

2D-DCT:  

![](lena_dct_2d.png)

2D-iDCT reconstruct image:  

![](lena_idct_2d.png)

## 1D-DCT

two 1D-DCT:  

![](lena_dct_1d.png)

two 1D-iDCT reconstruct image:

![](lena_idct_1d.png)

## Compare

2D-DCT time: 1981 sec  
2D-iDCT time: 2635 sec  
2D-iDCT PSNR: 28.23552  

1D-DCT time: 5sec  
1D-iDCT time: 5sec  
1D-iDCT PSNR: 274.56385

![](time_and_psnr.png)