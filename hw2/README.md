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

2D-DCT:  

using imshow():  
![](dct_2d.png)

using imwrite():  
![](lena_dct_2d.png)

2D-iDCT reconstruct image:  

![](lena_idct_2d.png)

two 1D-DCT:  

using imshow():  
![](dct_1d.png)

using imwrite():  
![](lena_dct_1d.png)
