%% sim data
dataset='lung';
N=9;
files=dir('datalung');

methods={'SL','SLNLM','GLSL','GLSL-BM3D'};
method=methods{3};
tic

for i =1:N
    file=files(i+2).name;
    fprintf('Running on : %s\n', file);
    if strcmp(dataset,'lung')
        load(['datalung/' file])
        dv=vi-vh;
        scale_v=sqrt(2.6037e-06^2/ mse(dv));
        dv=dv.*scale_v;
        width=41;
    end
    Jpixel=Y;
    norm_y = sqrt(mean(mean(dv.^2)));
    Jpixel1 = Jpixel/norm_y;
    dv = dv/norm_y;
 
    paramTable = {
        % method        alpha1      beta1       
        'GLSL',         0.1,       5;        
        'GLSL-BM3D',    0.1,        10;         
        'SL-NLM',       1e-3,       0;       
        'SL',           1e-3,       0;         
    };
    
    rowIdx = find(strcmp(paramTable(:, 1), method));
    
    if isempty(rowIdx)
        error('method: %s', method);
    end
    
    alpha1 = paramTable{rowIdx, 2};
    beta1  = paramTable{rowIdx, 3};
 

 
    pred=PnPEIT(Jpixel1,dv,'alpha1',  alpha1,'beta1',beta1,'verbose','yes', ...
        'IND',ind,'L',L,'WIDTH',width,'METHOD',method);%vi-vh
     

    reference=rescale(reference,-0.5,0.5);
    pred=rescale(pred,-0.5,0.5);
    

    % mkdir([dataset ,'/',method])
    % save([dataset ,'/',method,'/',file],'pred')
    rmse(i)=sqrt(mean((pred-reference).^2));
end
fprintf('average RMSE on LUNG dataset: %f\n', sum(rmse)/N);
toc

