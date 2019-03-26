lolfolder = '../dataset/jpegs/train';
fileFolder = fullfile(lolfolder,'*.jpeg');

qualityfactors = [75 85 95];
for qFactor = qualityfactors
    mkdir(lolfolder, int2str(qFactor));
    folder = fullfile(lolfolder, int2str(qFactor));
    mkdir(folder, 'mtr');
    mkdir(folder, 'ctr');
    mtrpath = fullfile(folder, 'mtr');
    ctrpath = fullfile(folder, 'ctr');
    mkdir(mtrpath, 'low');
    mkdir(ctrpath, 'low');
    mkdir(mtrpath, 'high');
    mkdir(ctrpath, 'high');
    mkdir(mtrpath, 'tonal');
    mkdir(ctrpath, 'tonal');
    mkdir(mtrpath, 'denoise');
    mkdir(ctrpath, 'denoise');
    mkdir(mtrpath, 'org');
    mkdir(ctrpath, 'org');
end
files = dir(fileFolder);
numfiles = length(files);

for k = 1:numfiles
  filepath = fullfile(files(k).folder, files(k).name);
  X = imread(filepath);
  [path,name,ext] = fileparts(filepath);

  % low pass
  
  l1 = imfilter(X,fspecial('gaussian',3,1),'symmetric');
  l2 = imfilter(X,fspecial('gaussian',5,1.5),'symmetric');
  l3 = imfilter(X,fspecial('average',3),'symmetric');

  l11 = down960(l1);
  l21 = down960(l2);
  l31 = down960(l3);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/low');
      writemtr = fullfile(folder, 'mtr/low');
      mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
      imwrite(l11, mtrpath, 'Quality',qFactor);
      mtrpath = fullfile(writemtr, sprintf('%sm2.jpeg', name));
      imwrite(l21, mtrpath, 'Quality',qFactor);
      mtrpath = fullfile(writemtr, sprintf('%sm3.jpeg', name));
      imwrite(l31, mtrpath, 'Quality',qFactor);
  end
  [lc1, lc2] = centrallyCrop(l11);
  [lc3, lc4] = centrallyCrop(l21);
  [lc5, lc6] = centrallyCrop(l31);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/low');
      writemtr = fullfile(folder, 'mtr/low');
  ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
  imwrite(lc1, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
  imwrite(lc2, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc21.jpeg', name));
  imwrite(lc3, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc22.jpeg', name));
  imwrite(lc4, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc31.jpeg', name));
  imwrite(lc5, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc32.jpeg', name));
  imwrite(lc6, ctrpath, 'Quality',qFactor);
  end

  % high pass
  
  l1 = imfilter(X,fspecial('unsharp',0.5),'symmetric');
  l2 = imsharpen(X,'Radius',1.5,'Amount',2);
  l3 = imsharpen(X,'Radius',2,'Amount',2);

  l11 = down960(l1);
  l21 = down960(l2);
  l31 = down960(l3);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/high');
  writemtr = fullfile(folder, 'mtr/high');
  mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
  imwrite(l11, mtrpath, 'Quality',qFactor);
  mtrpath = fullfile(writemtr, sprintf('%sm2.jpeg', name));
  imwrite(l21, mtrpath, 'Quality',qFactor);
  mtrpath = fullfile(writemtr, sprintf('%sm3.jpeg', name));
  imwrite(l31, mtrpath, 'Quality',qFactor);
  end
  
  [lc1, lc2] = centrallyCrop(l11);
  [lc3, lc4] = centrallyCrop(l21);
  [lc5, lc6] = centrallyCrop(l31);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/high');
  writemtr = fullfile(folder, 'mtr/high');
  ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
  imwrite(lc1, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
  imwrite(lc2, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc21.jpeg', name));
  imwrite(lc3, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc22.jpeg', name));
  imwrite(lc4, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc31.jpeg', name));
  imwrite(lc5, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc32.jpeg', name));
  imwrite(lc6, ctrpath, 'Quality',qFactor);
  end
  
  % denoising
  
  gray = rgb2gray(X);
  l1 = wiener2(gray,[3 3]);
  l2 = wiener2(gray,[5 5]);
  f = dbwavf('db12');
  l3 = imfilter(X, f, 'symmetric');

  l11 = down960(l1);
  l21 = down960(l2);
  l31 = down960(l3);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/denoise');
  writemtr = fullfile(folder, 'mtr/denoise');
  mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
  imwrite(l11, mtrpath, 'Quality',qFactor);
  mtrpath = fullfile(writemtr, sprintf('%sm2.jpeg', name));
  imwrite(l21, mtrpath, 'Quality',qFactor);
  mtrpath = fullfile(writemtr, sprintf('%sm3.jpeg', name));
  imwrite(l31, mtrpath, 'Quality',qFactor);
  end
  
  [lc1, lc2] = centrallyCrop(l11);
  [lc3, lc4] = centrallyCrop(l21);
  [lc5, lc6] = centrallyCrop(l31);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/denoise');
  writemtr = fullfile(folder, 'mtr/denoise');
  ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
  imwrite(lc1, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
  imwrite(lc2, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc21.jpeg', name));
  imwrite(lc3, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc22.jpeg', name));
  imwrite(lc4, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc31.jpeg', name));
  imwrite(lc5, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc32.jpeg', name));
  imwrite(lc6, ctrpath, 'Quality',qFactor);
  end

  % tonal adjustment
  l1 = imadjust(X,stretchlim(X,2/100),[],0.8);
  l2 = imadjust(X,stretchlim(X,6/100),[],1.2);
  l3 = histeq(X);

  l11 = down960(l1);
  l21 = down960(l2);
  l31 = down960(l3);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/tonal');
  writemtr = fullfile(folder, 'mtr/tonal');
  mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
  imwrite(l11, mtrpath, 'Quality',qFactor);
  mtrpath = fullfile(writemtr, sprintf('%sm2.jpeg', name));
  imwrite(l21, mtrpath, 'Quality',qFactor);
  mtrpath = fullfile(writemtr, sprintf('%sm3.jpeg', name));
  imwrite(l31, mtrpath, 'Quality',qFactor);
  end

  [lc1, lc2] = centrallyCrop(l11);
  [lc3, lc4] = centrallyCrop(l21);
  [lc5, lc6] = centrallyCrop(l31);
  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/tonal');
  writemtr = fullfile(folder, 'mtr/tonal');
  ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
  imwrite(lc1, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
  imwrite(lc2, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc21.jpeg', name));
  imwrite(lc3, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc22.jpeg', name));
  imwrite(lc4, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc31.jpeg', name));
  imwrite(lc5, ctrpath, 'Quality',qFactor);
  ctrpath = fullfile(writectr, sprintf('%sc32.jpeg', name));
  imwrite(lc6, ctrpath, 'Quality',qFactor);
  end

  % org image
  x11 = down960(X);
  [xc1, xc2] = centrallyCrop(x11);

  for qFactor = qualityfactors
      folder = fullfile(lolfolder, int2str(qFactor));
      writectr = fullfile(folder, 'ctr/org');
      writemtr = fullfile(folder, 'mtr/org');
      ctrpath = fullfile(writectr, sprintf('%sc1.jpeg', name));
      imwrite(xc1, ctrpath, 'Quality',qFactor);
      ctrpath = fullfile(writectr, sprintf('%sc2.jpeg', name));
      imwrite(xc2, ctrpath, 'Quality',qFactor);
      mtrpath = fullfile(writemtr, sprintf('%s.jpeg', name));
      imwrite(x11, mtrpath, 'Quality',qFactor);
  end
  fprintf('%d of %d done\n', k, numfiles);
end

trainval = ["val" "test"];
for i = trainval
    root = '../dataset/jpegs';
    lolfolder = fullfile(root, i);
    fileFolder = fullfile(lolfolder,'*.jpeg');
    for qFactor = qualityfactors
        mkdir(lolfolder, int2str(qFactor));
        folder = fullfile(lolfolder, int2str(qFactor));
        mkdir(folder, 'mtr');
        mkdir(folder, 'ctr');
        mtrpath = fullfile(folder, 'mtr');
        ctrpath = fullfile(folder, 'ctr');
        mkdir(mtrpath, 'low');
        mkdir(ctrpath, 'low');
        mkdir(mtrpath, 'high');
        mkdir(ctrpath, 'high');
        mkdir(mtrpath, 'tonal');
        mkdir(ctrpath, 'tonal');
        mkdir(mtrpath, 'denoise');
        mkdir(ctrpath, 'denoise');
        mkdir(mtrpath, 'org');
        mkdir(ctrpath, 'org');
    end
    files = dir(fileFolder);
    numfiles = length(files);

    for k = 1:numfiles
        filepath = fullfile(files(k).folder, files(k).name);
        X = imread(filepath);
        [path,name,ext] = fileparts(filepath);

        % low pass
        randomselect = randi(3);
        if randomselect==1
            l1 = imfilter(X,fspecial('gaussian',3,1),'symmetric');
        elseif randomselect==2
            l1 = imfilter(X,fspecial('gaussian',5,1.5),'symmetric');
        else
            l1 = imfilter(X,fspecial('average',3),'symmetric');
        end

        l11 = down960(l1);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/low');
            writemtr = fullfile(folder, 'mtr/low');
            mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
            imwrite(l11, mtrpath, 'Quality',qFactor);
        end
        
        [lc1, lc2] = centrallyCrop(l11);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/low');
            writemtr = fullfile(folder, 'mtr/low');
            ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
            imwrite(lc1, ctrpath, 'Quality',qFactor);
            ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
            imwrite(lc2, ctrpath, 'Quality',qFactor);
        end

        % high pass
        randomselect = randi(3);
        if randomselect==1
            l1 = imfilter(X,fspecial('unsharp',0.5),'symmetric');
        elseif randomselect==2
            l1 = imsharpen(X,'Radius',1.5,'Amount',2);
        else
            l1 = imsharpen(X,'Radius',2,'Amount',2);
        end

        l11 = down960(l1);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/high');
        writemtr = fullfile(folder, 'mtr/high');
        mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
        imwrite(l11, mtrpath, 'Quality',qFactor);
        end

        [lc1, lc2] = centrallyCrop(l11);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/high');
        writemtr = fullfile(folder, 'mtr/high');
        ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
        imwrite(lc1, ctrpath, 'Quality',qFactor);
        ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
        imwrite(lc2, ctrpath, 'Quality',qFactor);
        end

  
        % denoising
        gray = rgb2gray(X);
        randomselect = randi(3);
        if randomselect==1
            l1 = wiener2(gray,[3 3]);
        elseif randomselect==2
            l1 = wiener2(gray,[5 5]);
        else
            f = dbwavf('db12');
            l1 = imfilter(X, f, 'symmetric');
        end

        l11 = down960(l1);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/denoise');
        writemtr = fullfile(folder, 'mtr/denoise');
        mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
        imwrite(l11, mtrpath, 'Quality',qFactor);
        end

        [lc1, lc2] = centrallyCrop(l11);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/denoise');
        writemtr = fullfile(folder, 'mtr/denoise');
        ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
        imwrite(lc1, ctrpath, 'Quality',qFactor);
        ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
        imwrite(lc2, ctrpath, 'Quality',qFactor);
        end
        
        % tonal adjustment
        randomselect = randi(3);
        if randomselect==1
            l1 = imadjust(X,stretchlim(X,2/100),[],0.8);
        elseif randomselect==2
            l1 = imadjust(X,stretchlim(X,6/100),[],1.2);
        else
            l1 = histeq(X);
        end

        l11 = down960(l1);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/tonal');
        writemtr = fullfile(folder, 'mtr/tonal');
        mtrpath = fullfile(writemtr, sprintf('%sm1.jpeg', name));
        imwrite(l11, mtrpath, 'Quality',qFactor);
        end

        [lc1, lc2] = centrallyCrop(l11);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/tonal');
        writemtr = fullfile(folder, 'mtr/tonal');
        ctrpath = fullfile(writectr, sprintf('%sc11.jpeg', name));
        imwrite(lc1, ctrpath, 'Quality',qFactor);
        ctrpath = fullfile(writectr, sprintf('%sc12.jpeg', name));
        imwrite(lc2, ctrpath, 'Quality',qFactor);
        end

        % org image
        x11 = down960(X);
        [xc1, xc2] = centrallyCrop(x11);
        for qFactor = qualityfactors
            folder = fullfile(lolfolder, int2str(qFactor));
            writectr = fullfile(folder, 'ctr/org');
        writemtr = fullfile(folder, 'mtr/org');
        ctrpath = fullfile(writectr, sprintf('%sc1.jpeg', name));
        imwrite(xc1, ctrpath, 'Quality',qFactor);
        ctrpath = fullfile(writectr, sprintf('%sc2.jpeg', name));
        imwrite(xc2, ctrpath, 'Quality',qFactor);
        mtrpath = fullfile(writemtr, sprintf('%s.jpeg', name));
        imwrite(x11, mtrpath, 'Quality',qFactor);
        end
    end
end

