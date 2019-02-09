folder = '../dataset';
mkdir(folder, 'jpegs');
newpath = fullfile(folder, 'jpegs');
mkdir(newpath, 'train');
mkdir(newpath, 'val');
mkdir(newpath, 'test');
imgFolder = fullfile(folder,'*.data');
folders = dir(imgFolder);
numfolder = length(folders);
for j = 1:numfolder
    folder = fullfile(folders(j).folder, folders(j).name);
    fileFolder = fullfile(folder,'*.cr2');
    files = dir(fileFolder);

    numfiles = length(files);
    train = floor(numfiles*0.6);
    val = floor(numfiles*0.2);
    
    for k = 1:numfiles
        filepath = fullfile(files(k).folder, files(k).name);
        [path,name,ext] = fileparts(filepath);
        im = imread(filepath);
        q = unifrnd(85, 99);

        if k<train
            jpegwritepath = fullfile(newpath, 'train');
        elseif k<train+val
            jpegwritepath = fullfile(newpath, 'val');
        else
            jpegwritepath = fullfile(newpath, 'test');
        end

        imgpath = fullfile(jpegwritepath, sprintf('%s.jpeg', name));
        imwrite(im, imgpath, 'Quality', floor(q));
    end
end


