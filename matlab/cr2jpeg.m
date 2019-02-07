folder = 'dataset/raw/dataset';
fileFolder = fullfile(folder,'*.cr2');
files = dir(fileFolder);
numfiles = length(files);
for k = 1:numfiles
  filepath = fullfile(files(k).folder, files(k).name);
  [path,name,ext] = fileparts(filepath); 
  im = imread(filepath);
  jpegwritepath = fullfile(folder, 'jpegs');
  imgpath = fullfile(jpegwritepath, sprintf('%s.jpeg', name));
  q = unifrnd(85, 99);
  imwrite(im, imgpath, 'Quality', floor(q));
end