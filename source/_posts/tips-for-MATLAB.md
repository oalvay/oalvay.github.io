---
title: MATLAB related
date: 2019-07-03 14:50:47
tags:
---

some tips for MATLAB
<!-- more -->

#### basics

description|code|
---|---|---
not equal| ~=  
and | &&
or| II 
print| disp();
matrix| [row element; row element...]
speical matrix| zeros/ones/rand (row, column)<br> eye(size) for indentity matrix
sequence| start:interval:end
help| help xxx
size of matrix| size(matrix,*) <br>\*: 1--no of row <br> 2--no of column
length| length()
current directory| pwd
input file| load xxxx.xxx
output file | save xxxx.xxx b(variables);
clear all variables| clear
comment | %


#### Matrix manipulation

description|code|
---|---|---
select element of matrix| A([1 3], : )-- means exact all elements <br> from 1st & 3rd rows
add column| A = [A, [1; 2; 3]]
matrix to vector| A(:)
concatenating matrix| c = [A B] or c = [A; B]
multiplication| \* 
element operation| .+ .- .\* etc.
maximum| max()., note that max(matrix) gives max column
exact element with condition | eg. find(A < 3)
max element for rows/cols| max(A, [ ], 1 or 2)
reflect matrix | flipud()
inverse matrix| pinv()
transpose | transpose of a: a'

#### Plots
description|code|
---|---|---
multiple lines on a plot| hold on;
subplot: divide plot into b x c grid,<br>access dth element| subplot(b, c, d)

#### control statements

**while**:

```matlab
while condition  
    do something;  
end
```
**for**:

```matlab
for i= some series  
    do something;  
end
```

**if else**:

```matlab
if condition
    do something;
elseif condition
    do something;
else
    do something;
end
```
**function**:

```matlab
function [a, b] = undermat(x, y)
a = x^2;
b = a + y;
end
```


