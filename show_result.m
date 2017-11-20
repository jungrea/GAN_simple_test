for i= 1:1000:60000
    a = fin_output(i,:);
    a = reshape(a,28,28);
    imshow(imresize(a,20)',[]);
    pause(0.1);
end
