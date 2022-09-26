function w=Self_organizing_phase(data,n1,n2)
%Input
%data----data matrix, each row is one sample
%n1,n2----the no. of rows and columns of the 2D lattice of the neuron
%Output
%w----weights of the neurons
%
%No. of samples, dimensionality of input space, and total number of neurons
[nSample,nDim]=size(data);
nNeuron=n1*n2;
%pre-allocation for d which is used to fasten the code
d=zeros(nNeuron,1);
%Generate the initial weight vectors
w=randn(nNeuron,nDim);
%display the initial weight w
%disp(w);
%Define initial values for the time constants and the learning rate
eta0=0.1;
sigma0=sqrt((n1-1)^2+(n2-1)^2)/2;
tao1=1000/log(sigma0);

%Genetate the lateral distance matrix
%Dist=distance_matrix(n1,n2);
function Dist=distance_matrix(x1,x2)
% To calculate the distance between two neurons
Dist=(w(x1,:)-w(x2,:))*(w(x1,:)-w(x2,:))';
end
%The self-organizing phase
for k=1:100
    %calculate the learning rate and width of the neighbourhood function at
    %current iteration
    eta=eta0*exp(-k/tao1);
    sigma=sigma0*exp(-k/tao1);
    %randomly select a training sample
    %j is just a random number from 0 to nSample
    j=randperm(nSample,1);
    x=data(j,:);
    
    %compute and find the winning neuron
    %here d(i,1) is Euclidean distance square between w and x. d=sqrt(v*v')  
    for i=1:nNeuron
        d(i,1)=(w(i,:)-x)*(w(i,:)-x)';
    end
    %xx will get the value of min(d)->minimum distance value, index_win will get the row number
    [xx,index_win]=min(d);
    
    %update weight vector of all neurons
    for i=1:nNeuron
        h=exp(-distance_matrix(i,index_win)/2/sigma^2);
        w(i,:)=w(i,:)+eta*h*(x-w(i,:));
    end
end
end


    
    
