"""A module to specify vector index parameters for MilvusLikeClient"""
from enum import Enum
from typing import Union

class VecIndexType(Enum):
    """Vector index algorithm type"""
    GSIVFFLAT = 0
    GSDISKANN = 1

class IndexParam:
    """Vector index parameters.
    
    Attributes:
        index_name (string) : vector index name
        field_name (string) : vector index built on which field
        index_type (VecIndexType) :
            vector index algorithms (Only GSdisANN supported)
        kwargs :
            vector index parameters for different algorithms
    """
    DISKANN_ALGO_NAME = "gsdiskann"
    IVFFLAT_ALGO_NAME = "gsivfflat"

    def __init__(
        self, index_name: str, field_name: str, index_type: Union[VecIndexType, str], **kwargs
    ):
        self.index_name = index_name
        self.field_name = field_name
        self.index_type = index_type
        self.index_type = self._get_vector_index_type_str()
        self.metric_type = "l2"
        self.local_index = False
        self.kwargs = kwargs
        self.gs_param = self._parse_kwargs()


    def is_index_type_diskann_serial(self):
        return self.index_type in [
            IndexParam.DISKANN_ALGO_NAME
        ]
    
    def is_index_type_ivf_serial(self):
        return self.index_type in [
            IndexParam.IVFFLAT_ALGO_NAME,
        ]

    def _get_vector_index_type_str(self):
        """Parse vector index type to string."""
        if isinstance(self.index_type, VecIndexType):
            if self.index_type == VecIndexType.GSIVFFLAT:
                return IndexParam.IVFFLAT_ALGO_NAME
            elif self.index_type == VecIndexType.GSDISKANN:
                return IndexParam.DISKANN_ALGO_NAME
            raise ValueError(f"unsupported vector index type: {self.index_type}")
        assert isinstance(self.index_type, str)
        index_type = self.index_type.lower()
        if index_type not in [
            IndexParam.IVFFLAT_ALGO_NAME,
            IndexParam.DISKANN_ALGO_NAME,
        ]:
            raise ValueError(f"unsupported vector index type: {self.index_type}")
        return index_type

    def _parse_kwargs(self):
        # handle metric_type
        self.metric_type = self.kwargs.get('metric_type', 'l2')
        self.local_index = self.kwargs.get('local_index', False)
        print("local_index:", self.local_index)

        gs_params = {}
        params = self.kwargs.get('params', {})
        # handle param
        if self.is_index_type_ivf_serial():
            if 'ivf_nlist' in params:
                gs_params['nlist'] = params['ivf_nlist']
            if 'ivf_nlist2' in params:
                gs_params['nlist2'] = params['ivf_nlist2']
            if 'num_parallels' in params:
                gs_params['num_parallels'] = params['num_parallels']
            if 'enable_quantization' in params:
                gs_params['enable_quantization'] = params['enable_quantization']
            if 'quantization_type' in params:
                gs_params['quantization_type'] = params['quantization_type']

        elif self.is_index_type_diskann_serial():
            for param_name in [
                'pq_nseg',
                'pq_nclus',
                'queue_size',
                'num_parallels',
                'using_clustering_for_parallel',
                'lambda_for_balance',
                'enable_pq',
                'subgraph_count',
                'enable_neighbor_embedded',
                'enable_vector_copy',
                'build_with_quantized_vector',
                'quantization_type',
                'lvq_nclus',
                'graph_degree',
                'build_instruction_set',
                'params_adaptive'
            ]:
                if param_name in params:
                    gs_params[param_name] = params[param_name]
        return gs_params

    def param_str(self):
        """Parse vector index parameters to string."""
        gs_param = self.gs_param
        if gs_param == {}:
            return None
        partial_str = ",".join([f"{k}={v}" for k, v in gs_param.items()])
        return partial_str

    def __iter__(self):
        yield "field_name", self.field_name
        if self.index_type:
            yield "index_type", self.index_type
        yield "index_name", self.index_name
        yield from self.kwargs.items()

    def __str__(self):
        return str(dict(self))

    def __eq__(self, other: None):
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)

        if isinstance(other, dict):
            return dict(self) == other
        return False


class IndexParams:
    """Vector index parameters for MilvusLikeClient"""
    def __init__(self):
        self._indexes = {}

    def add_index(
        self, field_name: str, index_type: VecIndexType, index_name: str, **kwargs
    ):
        """Add `IndexParam` to `IndexParams`
        
        Args:
            field_name (string) : vector index built on which field
            index_type (VecIndexType) :
                vector index algorithms (Only DiskANN supported)
            index_name (string) : vector index name
            **kwargs: additional parameters for different index types
        """
        index_param = IndexParam(index_name, field_name, index_type, **kwargs)
        pair_key = (field_name, index_name)
        self._indexes[pair_key] = index_param

    def __iter__(self):
        yield from self._indexes.values()

    def __str__(self):
        return str(list(self))
