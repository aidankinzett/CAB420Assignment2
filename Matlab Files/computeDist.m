function dist = computeDist(v1, v2)
%compute distance from vecto1 to vector2
    dist = (v1(1) - v2(1))^2 + (v1(2) - v2(2))^2;
    dist = sqrt(dist);