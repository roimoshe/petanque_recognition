X = rgb2lab(RGB);

% Auto clustering
s = rng;
rng('default');
L = imsegkmeans(single(X),2,'NumAttempts',2);
rng(s);
BW = L == 2;

% Fill holes
BW = imfill(BW, 'holes');

% Invert mask
BW = imcomplement(BW);

% Create masked image.
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;