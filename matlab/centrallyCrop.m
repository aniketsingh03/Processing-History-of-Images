function [I1 I2] = centrallyCrop(I)
[row,col,~]=size(I);
if row>col && row>1024
    I1=I(floor((row-1024)/2)+1:floor((row-1024)/2)+512 ,floor((col-512)/2)+1:floor((col-512)/2)+512,:);
    I2=I(floor((row-1024)/2)+513:floor((row-1024)/2)+1024 ,floor((col-512)/2)+1:floor((col-512)/2)+512,:);
elseif col>row && col>1024
    I1=I(floor((row-512)/2)+1:floor((row-512)/2)+512 ,floor((col-1024)/2)+1:floor((col-1024)/2)+512,:);
    I2=I(floor((row-512)/2)+1:floor((row-512)/2)+512 ,floor((col-1024)/2)+513:floor((col-1024)/2)+1024,:);
else
    I1=I(1:512,1:512,:);
    I2=I(row-511:row,col-511:col,:);  
end
    