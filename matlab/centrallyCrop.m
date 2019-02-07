function [Im1 Im2] = centrallyCrop(I)
[h, w, dim] = size(I);
if w>h
    hi = floor(h/2) - 256;
    Im1=I(hi:hi+511, 1:512, :);
    Im2=I(hi:hi+511, w-511:w, :);
else 
   wi = floor(w/2) - 256;
   Im1=I(1:512, wi:wi+511, :);
   Im2=I(h-511:h, wi:wi+511, :);
end
    