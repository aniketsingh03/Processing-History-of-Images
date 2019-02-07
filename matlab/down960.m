function Im = down960(I)
[h, w, dim] = size(I);
hn = (h*960)/max(h,w);
wn = (w*960)/max(h,w);
hi = floor(unifrnd(min(h, hn), max(h, hn)));
wi = floor(unifrnd(min(w, wn), max(w,wn)));
Im = imresize(I, [hi wi]);