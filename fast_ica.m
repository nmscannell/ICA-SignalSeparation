img1 = imread('Recording-1.png');
img2 = imread('Recording-2.png');
n = 567*800;
x = [reshape(double(img1), [1, n]); ...
    reshape(double(img2), [1, n])];

% center data
mn = mean(x, 2);
x = x - repmat(mn, 1, n);

% whiten data
covariance = 1/(n-1)*(x*x');
[pc, v] = eig(covariance);
v = diag(v);
[junk, rindices] = sort(-1*v);
v = v(rindices);
pc = pc(:, rindices);
x = pc'*x;

% fast ica with exponential g function
g = @(x) x.*exp(-(x.^2)/2);
g_prime = @(x) (1-x.^2).*exp(-(x.^2)/2);

w = rand(2, 1);
w_opt = w;
threshold = 10^(-10);
while abs(dot(w, w_opt)-1) > threshold
    w = w_opt;
    w_opt = mean(x.*g(w'*x), 2) - mean(g_prime(w'*x)).*w;
    w_opt = w_opt/norm(w_opt);
end

s = w_opt.*x;
s = s + w_opt'*mn;
s1 = reshape(s(1,:), [567,800]);
s2 = reshape(s(2,:), [567,800]);
imwrite(mat2gray(s1), './s1.png');
imwrite(mat2gray(s2), './s2.png');