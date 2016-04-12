classdef multiLayerPerceptron < handle
    %MULTILAYERPERCEPTRON for regression or binary classification
    % implementation of multi-layer perceptron (MLP),  also known as 
    % feed-forward neural network
    
    properties
        X       % input data
        y       % output data
        N       % number of data samples
        D       % data dimensionality
        W       % model weights
        grad    % current gradients
        a       % current hidden variable states
        z       % current non-linear hidden variable states
        layers  % number of network layers
        h       % activation function
        dh      % derivative of activation function
        L       % loss function  
        dL      % derivative of loss function
    end
    
    methods
        function obj = multiLayerPerceptron(X, y, hiddens, activation, mode)
            % initialise data
            obj.X = X;
            obj.y = y;
            [obj.N, obj.D] = size(X);
            % initialise network
            hiddens = [hiddens, 1];
            obj.layers = length(hiddens);
            for i = 1 : obj.layers
                obj.a{i} = rand(hiddens(i), 1);
                obj.z{i} = zeros(hiddens(i), 1);
            end
            network = [obj.D, hiddens];
            % initialise parameters
            for i = 1 : length(network) - 1
                obj.W{i} = zeros(network(i + 1), network(i));
                obj.grad{i} = zeros(network(i + 1), network(i));
            end
            % initialise activation function and derivative
            if strcmp(activation, 'tanh')
                obj.h = @(x) tanh(x);
                obj.dh = @(x) 1 - tanh(x).^2;
            elseif strcmp(activation, 'sigma')
                obj.h = @(x) sigma(x);
                obj.dh = @(x) dSigma(x);
            else
                error('Invalid activation function');
            end
            % set cost functions
            if strcmp(mode, 'regression')
                obj.L = @(y, yhat) (y - yhat)^2;
                obj.dL = @(y, z) -2 * (y - z)';
            elseif strcmp(mode, 'classification')
                obj.L = 1;
                obj.dL = 1;
            else
                error('Invalid mode');
            end
        end
        function forwardPropagate(obj, index)
            % update hidden variables with forward propagation
            obj.a{1} = obj.W{1} * obj.X(index, :)';
            obj.z{1} = obj.h(obj.a{1});
            % update interior layers
            for i = 2 : obj.layers
                obj.a{i} = obj.W{i} * obj.z{i - 1};
                obj.z{i} = obj.h(obj.a{i});
            end
        end
        function backPropagate(obj, index, alpha)
            % compute gradients with back propagation
            K = obj.layers;
            del = diag(obj.dh(obj.a{K})) * obj.dL(obj.y(index), obj.z{K});
            obj.grad{obj.layers} = del * obj.z{obj.layers - 1}';
            obj.updateWeights(alpha, obj.layers);
            % compute interior gradients
            for i = obj.layers - 1 : -1 : 2
                del = diag(obj.dh(obj.a{i})) * obj.W{i + 1}' * del;
                obj.grad{i} = del * obj.z{i - 1}';
                obj.updateWeights(alpha, i);
            end
            % compute base gradient
            del = diag(obj.dh(obj.a{1})) * obj.W{2}' * del;
            obj.grad{1} = del * obj.X(index, :);
            obj.updateWeights(alpha, 1);
        end
        function updateWeights(obj, alpha, i)
            % perform (stochastic) gradient descent step on weights
            obj.W{i} = gradientDescent(alpha, obj.W{i}, obj.grad{i});
        end
        function train(obj, maxIters, alpha)
            % learn model weights
            iters = 0;
            while iters < maxIters
                cost = 0;
                % randomly permute data
                idx = randperm(obj.N);
                obj.X = obj.X(idx, :);
                obj.y = obj.y(idx, :);
                % run over epoch
                for i = 1 : obj.N
                    obj.forwardPropagate(i);
                    obj.backPropagate(i, alpha);
                    yhat = obj.predict(obj.X(i, :));
                    cost = cost + obj.L(obj.y(i), yhat);
                end
                fprintf('\rCost: %f\n', cost / obj.N);
                iters = iters + 1;
            end
        end
        function yhat = predict(obj, xhat)
            % make prediction by propagating forwards
            yhat = obj.h(obj.W{1} * xhat');
            for i = 2 : obj.layers
                yhat = obj.h(obj.W{i} * yhat);
            end
%             if strcmp(obj.mode, 'classification')
%                 yhat = sigma(yhat);
%             end
        end
    end
end
