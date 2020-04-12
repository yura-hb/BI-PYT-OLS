class Vertex:
    
    id = 0
    
    def __init__(self):
        self.id = self.id_next()
        self.neighbours = set()
    
    @classmethod
    def id_next(cls):
        cls.id += 1
        return cls.id
    
    def add_neighbours(self, *vertices):
        for vertex in vertices:
            assert isinstance(vertex, Vertex), "Zadáno '" + str(vertex) + "' při přidávání souseda, očekáván typ 'Vertex'."
            vertex.neighbours.add(self)
        self.neighbours.update(vertices)
            
    def remove_neighbours(self, *vertices):
        for vertex in vertices:
            assert isinstance(vertex, Vertex), "Zadáno '" + str(vertex) + "' při přidávání souseda, očekáván typ 'Vertex'."
            self.neighbours.discard(vertex)
        for vertex in vertices:
            vertex.neighbours.discard(self)
    
    @property
    def degree(self):
        return len(self.neighbours)
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __str__(self):
        return 'Vrchol číslo ' + str(self.id) + ' má sousedy ' + str(sorted(self.neighbours))
    
    def __repr__(self):
        return str(self.id)
    
