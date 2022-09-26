function w=Convergence_phase(data,w,n1,n2)
%The convergence phase
%set the learning rate to a samall constant
eta=0.01;
nNeuron=n1*n2;
[nSample,nDim]=size(data);
d=zeros(nNeuron,1);
%repeat 500*nNeuron times
for k=1:500*nNeuron
    %randomly select a sample
    j=randperm(nSample,1);
    x=data(j,:);
    
    % compete and find the winning neuron
    for i=1:nNeuron
        d(i,1)=(w(i,:)-x)*(w(i,:)-x)';
    end
    %xx will get the value of min(d), index_win will get the row number
    [xx,index_win]=min(d);
    
    % update the weight vector of the winning neuron only
    h=1;
    w(index_win,:)=w(index_win,:)+eta*h*(x-w(index_win,:));
end
