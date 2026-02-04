%% sim data
dataset='sim';
N=20;
files=dir('data');

methods={'SL','SL-NLM','GLSL','GLSL-BM3D'};
method=methods{3};
tic
predd=[];
for i =1:N 
    file=files(i+2).name;
    fprintf('Running on : %s\n', file);
    load(['data/' file])
    width=39;


    norm_y = sqrt(mean(mean(dv.^2)));
    Jpixel1 = Jpixel/norm_y;
    dv = dv/norm_y;
 
    paramTable = {
    % method      alpha1   beta1
    'GLSL',       0.25,    0.25;
    'GLSL-BM3D',  0.25,    0.25;   
    'SL-NLM',     0.15,    0;       
    'SL',         0.1,     0;       
    };
    
    rowIdx = find(strcmp(paramTable(:, 1), method));
    
    if isempty(rowIdx)
        error('unknown method: %s', method);
    end
    
    alpha1 = paramTable{rowIdx, 2};
    beta1  = paramTable{rowIdx, 3};


    
    pred=PnPEIT(Jpixel1,dv,'alpha1',  alpha1,'beta1',beta1,'verbose','yes', ...
        'IND',ind,'L',L,'WIDTH',width,'METHOD',method );
    predd=[predd,pred];
    % mkdir([dataset ,'/',method])
    % save([dataset ,'/',method,'/',file],'pred')
    rmse(i)=sqrt(mean((pred-reference).^2));
end
fprintf('average RMSE on EIDORS dataset: %f\n', sum(rmse)/N);
toc


down=-0.5;up=0.5;step=0.2;
figure
for i=1:4
    for j=1:5
        subplot(4,5,(i-1)*5+j)
        get_recplot(xc,yc,  predd(:,(i-1)*5+j),down,up,step);
    end
end
