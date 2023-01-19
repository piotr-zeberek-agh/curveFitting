# Curve Fitting Project
Polynomial regression using ordinary least square estimation.

## Work status
Everything works.

## How to run
Just run *run.sh* script in terminal. 
```
./run.sh
```
Default execution mode is demo. If you want to specify file to read data from and order of polynomial just run as below.
```
./run.sh <file> <order>
```
*data.txt* contains 1024 data points generated with 8th order polynomial with coefficients from B<0> to B<8> 
```
 7.4  -3.31  2.11  0.78  -0.4  -5  2.93  -0.001  0.087
```
and *x* ranging from -5 to 5.

For each *y* value noise was added.
```
noise = random value between +-0.005*y
```

## Possible future additions
* [DONE] Reading (x,y) pairs from data file. (fairly easy)
* [DONE for multiplication and tranposition] Integrating tiled algorithms. (harder)
* [GAVE UP] Rewriting code as a cpu version for comparison purposes. (quite hard and time consuming)
