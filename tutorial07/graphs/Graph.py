class Graph:
    
    id = 0
    
    def __init__(self, *vertices):
        self.id = self.id_next()
        self.vertices = set(vertices)
    
    @classmethod
    def id_next(cls):
        cls.id += 1
        return cls.id
    
    def add_vertices(self, *vertices):
        self.vertices.update(vertices)
        
    def remove_vertices(self, *vertices):
        for vertex in vertices:
            vertex.remove_neighbours(*vertex.neighbours)
            self.vertices.discard(vertex)

    def connect_vertices(self, vertex1, vertex2):
        vertex1.add_neighbours(vertex2)
      
    @property
    def vertex_number(self):
        return len(self.vertices)
    
    @property
    def edges(self):
        edges = set()
        for vertex in self.vertices:
            for neighbour in vertex.neighbours:
                edges.add( frozenset((vertex.id, neighbour.id)) )
        return edges
    
    @property
    def export_graph(self):
        s = str(self.vertex_number) + ';'
        s += ','.join( '-'.join(str(idx) for idx in edge) for edge in self.edges )
        return s
    
    def import_graph(self, xs):
        vertex_number, edges = xs.split(';')
        edges = [tuple(e.split('-')) for e in edges.split(',')]
        vertices = set(e[0] for e in edges).union(set(e[1] for e in edges))
        vertices = {idx: Vertex() for idx in vertices}
        
        for idx, vertex in vertices.items():
            vertex.id = idx
        
        self.add_vertices(*list(vertices.values()))
        for vertex_a, vertex_b in edges:
            self.connect_vertices(vertices[vertex_a], vertices[vertex_b])
     
    def __str__(self):
        return 'Graf číslo ' + str(self.id) + ' obsahuje vrcholy ' + str(sorted(self.vertices)) + ' spojené hranami ' + str(self.edges)
    
