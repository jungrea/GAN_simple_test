clc
clear
%% 构造真实训练样本 60000个样本 1*784维（28*28展开）
load mnist_uint8;

train_x = double(train_x(1:60000,:)) / 255;
% 真实样本认为为标签 1； 生成样本为0;
train_y = double(ones(size(train_x,1),1));
% normalize
train_x = mapminmax(train_x, 0, 1);

%% 构造模拟训练样本 60000个样本 1*100维
test_x = normrnd(0,1,[60000,100]); % 0-255的整数
test_x = mapminmax(test_x, 0, 1);

test_y = double(zeros(size(test_x,1),1));
test_y_rel = double(ones(size(test_x,1),1));

%%
nn_G_t = nnsetup([100 784]);
nn_G_t.activation_function = 'sigm';
nn_G_t.output = 'sigm';

nn_D = nnsetup([784 100 1]);
nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn_D.dropoutFraction = 0.5;   %  Dropout fraction 
nn_D.learningRate = 0.01;                %  Sigm require a lower learning rate
nn_D.activation_function = 'sigm';
nn_D.output = 'sigm';
% nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay

nn_G = nnsetup([100 784 100 1]);
nn_G.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn_G.dropoutFraction = 0.5;   %  Dropout fraction 
nn_G.learningRate = 0.01;                %  Sigm require a lower learning rate
nn_G.activation_function = 'sigm';
nn_G.output = 'sigm';
% nn_G.weightPenaltyL2 = 1e-4;  %  L2 weight decay

opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples
%%
num = 1000;
D_num = 2; % D每次训练的次数
G_num = 1; % G每次训练的次数
tic
for each = 1:2000
    disp(['each ======================',num2str(each)]);
    %----------计算G的输出：假样本------------------- 
    for i = 1:length(nn_G_t.W)   %共享网络参数
        nn_G_t.W{i} = nn_G.W{i};
    end
    G_output = nn_G_out(nn_G_t, test_x);
    %-----------训练D------------------------------
    for k1 = 1:D_num
        index = randperm(60000);
        train_data_D = [train_x(index(1:num),:);G_output(index(1:num),:)];
        train_y_D = [train_y(index(1:num),:);test_y(index(1:num),:)];
        nn_D = nntrain(nn_D, train_data_D, train_y_D, opts);%训练D
    end
    %-----------训练G-------------------------------
    for i = 1:length(nn_D.W)  %共享训练的D的网络参数
        nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
    end
    for k2 = 1:G_num
        %训练G：此时假样本标签为1，认为是真样本
        nn_G = nntrain(nn_G, test_x(index(1:num),:), test_y_rel(index(1:num),:), opts);
    end
end
toc
for i = 1:length(nn_G_t.W)
    nn_G_t.W{i} = nn_G.W{i};
end
fin_output = nn_G_out(nn_G_t, test_x);

%% 可视化结果：挑选一部分显示
for i= 1:1000:60000
    a = fin_output(i,:);
    a = reshape(a,28,28);
    imshow(imresize(a,20)',[]);
    pause(0.1);
end



