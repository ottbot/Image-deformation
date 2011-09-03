function vec = pngboundary(name, vsize)


    picname = sprintf('%s.png', name);

    [X, map] = imread(picname);
    
    
    %# Check what type of image it is and convert to grayscale:
    
    if ~isempty(map)                %# It's an indexed image if map isn't empty
        grayImage = ind2gray(X,map);  %# Convert the indexed image to grayscale
    elseif ndims(X) == 3            %# It's an RGB image if X is 3-D
        grayImage = rgb2gray(X);      %# Convert the RGB image to grayscale
    else                            %# It's already a grayscale or binary image
        grayImage = X;
    end
    
    if islogical(grayImage)         %# grayImage is already a binary image
        BW = grayImage;
    else
        level = graythresh(grayImage);     %# Compute threshold
        BW = im2bw(grayImage,level);  %# Create binary image
    end
    
    BW = flipud(BW);
    %imshow(BW)
    
    [B,L,N] = bwboundaries(BW);
    
    for k=1:length(B),
        boundary = B{k};
        x = boundary(:,2);
        y = boundary(:,1);
    end
    
    
    
    x = smooth(x); %,'sgolay',2)
    y = smooth(y); %,'sgolay',2)
    

    
    xn = linspace(min(x),max(x), size(x,1))';
    xi = linspace(min(x),max(x), vsize);
    
    x = interp1(xn,x,xi);
    
    
    yn = linspace(min(y),max(y), size(y,1))';
    yi = linspace(min(y),max(y), vsize);
    
    y = interp1(yn,y,yi);
    
    txtname = sprintf('%s.txt', name);
    
    %figure()
    %imshow(BW)
    figure()
    plot(x,y)
    
    vec = [x y];
    dlmwrite(txtname, vec','precision', '%10.6f');
    

end