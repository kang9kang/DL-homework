import numpy as np

class MyTensor(np.ndarray):
    def __new__(cls, data, requires_grad=True):
        obj = np.asarray(data).view(cls)
        obj.requires_grad = requires_grad
        obj.grad = np.zeros_like(obj.view(np.ndarray)) if requires_grad else None
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.requires_grad = getattr(obj, 'requires_grad', True)
        self.grad = getattr(obj, 'grad', None)
        
    def __str__(self) -> str: 
        return f"Tensor({self.view(np.ndarray)}, requires_grad={self.requires_grad})"
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        requires_grad = False
        for _, input_ in enumerate(inputs):
            if isinstance(input_, MyTensor):
                requires_grad = requires_grad or input_.requires_grad
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)
        
        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for _, output in enumerate(outputs):
                if isinstance(output, MyTensor):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout
            
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        
        if results is NotImplemented:
            return NotImplemented
        
        if isinstance(results, tuple):
            results = tuple(MyTensor(r, requires_grad=requires_grad) if isinstance(r, np.ndarray) else r for r in results)
        else:
            results = MyTensor(results, requires_grad=requires_grad)
            
        return results
        