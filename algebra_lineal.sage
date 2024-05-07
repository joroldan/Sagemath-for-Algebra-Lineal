import re

# Práctica 1

def matrix_change_basis(original, final=None, field=None):
    """Calcula la matriz de cambio de base entre una base original y una base final.
    
    Parámetros:
        - Base original: lista de vectores linealmente independientes
        - Base final: lista de vectores linealmente independientes (si no se da se asume la canónica)
        - Cuerpo: si no se da se asume el mismo en el que viven los vectores de la base original
        
    Ejemplo 1:

    |    v1 = vector(QQ,[1,2,3])
    |    v2 = vector(QQ,[1,1,3])
    |    v3 = vector(QQ,[1,0,1])
    |    baseSalida = [v1,v2,v3]
    |    matrix_change_basis(baseSalida)
    
    Ejemplo 2:

    |    v1 = vector(QQ,[1,2])
    |    v2 = vector(QQ,[1,1])
    |    baseSalida  = [v1,v2]
    |    w1 = vector(QQ,[0,1])
    |    w2 = vector(QQ,[1,0])
    |    baseLlegada = [w1,w2]
    |    matrix_change_basis(baseSalida, baseLlegada)
    """
    if field is None:
        field = original[0].base_ring()

    if final is None:
       V = VectorSpace(field, len(original[0]))
    else:
        V = VectorSpace(field, len(final))
        V = V.subspace_with_basis(final)
        
    return matrix([V.coordinates(elto) for elto in original]).transpose()

def first_non_null(v):
    """Dado un vector v te devuelve la posición del primer no elemento nulo o -1 si todos son nulos"""
    for i in range(len(v)):
        if (v[i]!=0):
            return i 
    return -1

def pivot_columns(m):
    """Dada una matriz m te dice en qué columnas aparecen pivotes al escalonarla"""
    m = m.echelon_form()
    return [first_non_null(row) for row in m.rows() if first_non_null(row) != -1]

def non_pivot_columns(m):
    """Dada una matriz m te dice en qué columnas no aparecen pivotes al escalonarla"""
    total  = set(range(m.ncols()))
    pivots = set(pivot_columns(m))
    return list(total.difference(pivots))

def complement(s1, s2):
    """Dados dos subespacios s1 ≤ s2 calcula un s3 tal que s1 ⊕ s3 = s2"""
    if not(s1.is_subspace(s2)):
        raise ValueError("First parameter is not a subspace of the second one") 
    b1 = s1.basis_matrix().transpose()
    b2 = s2.basis_matrix().transpose()
    m  = b1.augment(b2)
    np = pivot_columns(m)
    n  = s1.dimension()
    return span(s1.base_ring(),[s2.basis()[i-n] for i in np if i >= n])

def projections(v, subspace1, subspace2): # Calcula las dos proyeccion de V en S1 y S2
    """Dado un vector v ∈ subspace1 ⊕ subspace2 calcula las dos proyecciones sobre cada uno de ellos
    
    Ejemplo:
    
    |    v1 = vector(QQ,[1,0,1])
    |    v2 = vector(QQ,[0,1,1])
    |    v3 = vector(QQ,[1,2,1])
    |    S1 = span([v1])
    |    S2 = span([v2,v3])
    |    v  = vector(QQ,[1,2,3])
    |    p1, p2 = projections(v, S1, S2)
    """
    b1 = list(subspace1.basis())
    b2 = list(subspace2.basis())
    V  = subspace1 + subspace1.complement()
    V  = V.subspace_with_basis(b1 + b2)
    c  = V.coordinates(v)
    n  = subspace1.dimension()
    p1 = subspace1.linear_combination_of_basis(c[:n])
    p2 = subspace2.linear_combination_of_basis(c[n:])
    return p1, p2

def multi_union(*args):
    """Permite unir más de dos listas eliminando repetidos"""
    result = set
    for v in args:
        result = result.union(v)
    return list(result)
    
def listvariables(l):
    """Calcula la lista de variables que forman parte de una lista l"""
    vars = set()
    for i in l:
        try:
            vars = vars.union(i.variables())
        except AttributeError:
            pass
    return list(vars)
    
def coordinates(vectorspace, v):
    """Método expandido de vectorspace.coordinates() para trabajar con variables
    
    Ejemplo:
    
    |    v1 = vector(QQ,[1,0,1])
    |    v2 = vector(QQ,[0,1,1])
    |    v3 = vector(QQ,[1,2,1])
    |    V  = VectorSpace(QQ,3).subspace_with_basis([v1,v2,v3])
    |    var('x')
    |    coordinates(V,[1,2*x+2,3*x-1])

    Si se aplica sobre un espacio que no es el total muestra un warning de qué necesita el vector para estar contenido en el espacio.
    """
    variables = listvariables(v) #multi_union(*map(lambda e: e.variables(), v))
    p = PolynomialRing(vectorspace.base(), variables) 
    m = vectorspace.basis_matrix().transpose()
    m = m.base_extend(p)
    m = m.augment(vector(p, v))
    coords = m.echelon_form().column(-1)
    coord1 = coords[0:vectorspace.dimension()]
    coord2 = list(filter(lambda a: a!=0, coords[vectorspace.dimension():]))
    if len(coord2) > 0:
        fun = warnings.formatwarning
        warnings.formatwarning = lambda a,b,c,d,e='': str(f"Warning: {a}")
        if len(coord2) == 1:
            warnings.warn(f"{coord2[0]} must be zero.")
        else:
            warnings.warn(f"{coord2} must be all zero.")
        warnings.formatwarning = fun
    return coord1

def coordinate(vectorspace, v, i):
    """Método similar a coordinates(vectorspace, v) pero devolviendo la i-ésima coordenada (empezando a contar en cero)"""
    return coordinates(vectorspace, v)[i]

def scalar(u,v):
    """Producto escalar que devuelve el resultado como vector de longitud 1"""
    return vector(u.base_ring(),[u.dot_product(v)])

def annihilator_basis_to_equations(basis, variables):
    """Convierte lo que devuelve "annihilator_basis" en ecuaciones usando las variables indicadas.
    
    Ejemplo:
    
    |    aB = V.annihilator_basis([matrix(QQ,[1,0,2]).transpose()])
    |    var('x y z')
    |    annihilator_basis_to_equations(aB, (x,y,z))
    """
    return [sum([elto[i] * variables[i] for i in range(len(variables))]) == 0 for elto in basis]

def annihilator_basis_to_expressions(basis, variables):
    """Similar a "annihilator_basis_to_equations" pero sin igualar las exrpresiones a cero."""
    return matrix(basis)*vector(variables)

def equations_to_matrix(eqns, variables, field=None):
    """Dada una lista de ecuaciones y una lista de variables devuelve la matriz de coeficientes del sistema"""
    if field is None:
        field = eqns[0].lhs().base_ring()
    return matrix(field,[[eq.lhs().coefficient(v) - eq.rhs().coefficient(v) for v in variables] for eq in eqns])

def equations_to_vector_of_independent_terms(eqns, variables, field=None):
    """Dada una lista de ecuaciones y una lista de variables devuelve el vector de términos independientes del sistema"""
    m = equations_to_matrix(eqns, variables, field)
    field = m.base_ring()
    v = vector(field,[eq.rhs() - eq.lhs() for eq in eqns]) +  m * vector(variables)
    return v

def equations_to_augmented_matrix(eqns, variables, subdivide=True, field=None):
    """Dada una lista de ecuaciones y una lista de variables devuelve la matriz ampliada de coeficientes del sistema. Por defecto con subdivisiones"""
    m = equations_to_matrix(eqns, variables, field)
    v = equations_to_vector_of_independent_terms(eqns, variables, field)
    return m.augment(v,subdivide)

def create_chain(xlen,ylen,vsep,x0,y0,num):
    """Método auxiliar para crear el diagrama de show_box_figure.

    Parámetros:
     - xlen: anchura de la caja
     - ylen: altura de la caja
     - vsep: separación entre cajas
     - x0: coordenada x inicial de la caja inferior
     - y0: coordenada y inicial de la caja inferior
     - num: número de cajas a construir
    """
    polT = polygon2d([[x0,y0], [x0,y0+ylen], [x0+xlen,y0+ylen], [x0+xlen,y0]], fill=False, thickness=1, color='black')
    for i in range(1,num):
        arr   = arrow(*map(lambda v: [v[0],v[1]+vsep*(i-1)+ylen*i], [[x0+xlen/2,y0+vsep],[x0+xlen/2,y0]]), color='black', arrowsize=2, width=1)
        pol   = polygon2d(map(lambda v: [v[0],v[1]+(vsep+ylen)*i], [[x0,y0], [x0,y0+ylen], [x0+xlen,y0+ylen], [x0+xlen,y0]]), fill=False, thickness=1, color='black')
        polT += arr + pol
    return polT  

def box_structure(m, field = None):
    """Dada una matriz m y un cuerpo (opcional, si no se toma el de la matriz) calcula la estructura por cadenas de bloques de la matriz. Devuelve así un diccionario cuyas claves son las potencias de los polinomios de la factorización del pol. característico y toma por valores parejas: [[numero de cadenas de longitud i para i=1,...,k], [ker(p(f)^i) para i=0,...,k]]. Este diccionario recibe el nombre de estructura y es la entrada de otros muchos métodos"""
    if field is None:
        field = m.base_ring()
    p.<x> = PolynomialRing(field)
    pol = p(m.characteristic_polynomial())
    lis = list(factor(pol))
    dib = {elto[0]: [] for elto in lis}
    for elto in lis:
        m2 = matrix(field,(elto[0]).subs(x=m))
        spaces = [(m2^n).right_kernel() for n in (0..elto[1]+1)]
        dims   = [space.dimension() for space in spaces]
        dib[elto[0]] = [[(2*dims[i] - dims[i-1] - dims[i+1])/elto[0].degree() for i in (1..elto[1])], spaces]
    return dib
    
def box_chain_lengths_number(m, field = None):
    """Dada una matriz calcula el número de cadenas de bloques que hay de cada longitud"""
    struc = box_structure(m, field)
    return box_chain_lengths_number_from_structure(struc)

def box_chain_lengths_number_from_structure(struc):
    """Dada una estructura (ver "box_structure") calcula el número de cadenas de bloques que hay de cada longitud"""
    dib = {key: struc[key][0] for key in struc.keys()}
    return dib
    
def show_box_figure(m, field = None):
    """Dada una matriz representa mediante un diagrama las distintas cadenas de bloques"""
    chains = list()
    pos = 0
    dib = box_chain_lengths_number(m, field)
    for key in dib.keys():
        for i in (1..len(dib[key])):
            for j in (1..dib[key][i-1]):
                chains.append(create_chain(1,1,0.5,pos,0,i) + text(key,(pos+0.5,-0.25),color='black'))
                pos += 1.5
        pos += 1
    show(sum(chains),axes=False)

def length_of_chains(m, field = None):
    """Dada una matriz calcula las longitudes de las distintas cadenas de bloques"""
    dimen = box_chain_lengths_number(m, field)
    return length_of_chains_from_chain_lengths_number(dimen)

def length_of_chains_from_chain_lengths_number(dimen):
    """Dado el resultado de "box_chain_lengths_number_from_structure/box_chain_lengths_number" calcula las longitudes de las distintas cadenas de bloques"""
    dimen2 = {}
    for key in dimen.keys():
        deg = key.degree()
        v = list()
        for ind, l in enumerate(dimen[key]):
            if l != 0:
                for i in range(l):
                    v.append(ind+1)
        v.reverse()
        dimen2[key] = v
    return dimen2
    
def box_heads(m, field = None):
    """Dada una matriz m da una posible elección de los vectores cabeza de cada cadena de bloques prececidos por la longitud en bloques de la cadena que encabezan"""
    if field is None:
        field = m.base_ring()
    struc = box_structure(m, field)
    dimen = box_chain_lengths_number_from_structure(struc)
    chain = length_of_chains_from_chain_lengths_number(dimen)
    p.<x> = PolynomialRing(field)
    heads = {}
    for key in chain.keys():
        deg = key.degree()
        known_spaces = [span([vector(field,[0]*(m.ncols()))])]*max(chain[key])
        kerne_spaces = struc[key][1]
        heads[key] = []
        for i in chain[key]:
            candidates = complement(kerne_spaces[i-1] + known_spaces[i-1], kerne_spaces[i])
            v = candidates.basis()[0]
            heads[key] += [[i,v]]
            for j in [0..i-1]:
                known_spaces[j]+= span([(key.subs(x=m))^(i-1-j)*(m^k)*v for k in [0..deg-1]])
    return heads

def box_matrix(m, field = None, big_blocks = False, transformation = False, subdivisions = True, 
                reverseChainOrder = False, reverseCompanionOrder = False, reverseOrder = None):
    """
    Dada una matriz devuelve su matriz por bloques.
    
    Parámetros:
     - m es la matriz inicial
     - field es el cuerpo de trabajo, si no se da ninguno se toma el de la matriz
     - big_blocks (por defecto False) si lo ponemos a True toma la cadena toda unida como una única compañera de su cabeza
     - transformation (por defecto False) si lo ponemos a True, devuelve no solo la matriz por bloques B sino también P tal que PBP⁻¹ = A
     - subdivisions (por defecto True) pone o no las líneas de corte en cada cadena
     - reverseChainOrder (por defecto False) toma como base la cadena de abajo a arriba
     - reverseCompanionOrder (por defecto False) toma como base en cada compeñera el orden normal o el orden inverso:  {fⁿ⁻¹(v), fⁿ⁻²(v), ..., f²(v), f(v), v}
     - reverseOrder (por defecto None) si se le da valor, este valor pisa a los dos anteriores parámetros
     
    Ejemplo:
    
    |   B, P = box_matrix(A, big_blocks=True, transformation=True)
    |   show(B, P)
    |   print(P*B*~P == A)
    """
    if field is None:
        field = m.base_ring()
    if reverseOrder is not None:
        reverseChainOrder = reverseOrder
        reverseCompanionOrder = reverseOrder
    heads  = box_heads(m, field)
    basis  = list()
    blocks = [0]
    for key in heads.keys():
        deg = key.degree()
        for pair in heads[key]:
            blocks.append(blocks[-1] + deg*pair[0])
            if big_blocks:
                start, end, step = 0, deg*pair[0], 1
                if reverseCompanionOrder:
                    start, end, step = end-1, start-1, -1
                basis += [(m^i)*pair[1] for i in range(start,end,step)]
            else:
                m2 = key.subs(x=m)
                startJ, endJ, stepJ = 0, pair[0], 1
                startI, endI, stepI = 0, deg, 1
                if reverseCompanionOrder:
                    startI, endI, stepI = endI-1, startI-1, -1
                if reverseChainOrder:
                    startJ, endJ, stepJ = endJ-1, startJ-1, -1
                basis += [(m2^j)*(m^i)*pair[1] for j in range(startJ,endJ,stepJ) for i in range(startI,endI,stepI)]
    chgm = matrix(field,len(basis),basis).T
    newm = ~chgm * m * chgm
    if subdivisions:
        newm.subdivide(blocks[1:-1],blocks[1:-1])
    if transformation:
        return newm, chgm
    else:
        return newm
    
def constant_matrix(number, rows, columns = None):
    """Crea una matriz de tamaño rows x columns o rows x rows rellena con el valor number"""
    if columns is None:
        columns = rows
    return matrix([[number for i in (1..columns)] for j in (1..rows)])

def link_matrix_down(number1, number2, size):
    """Crea una matriz cuadrada size x size rellena con el valor number2 salvo en la esquina superior derecha donde hay una celda con el valor number1"""
    return matrix([[number1 if i == size and j==1 else number2 for i in (1..size)] for j in (1..size)])

def link_matrix_up(number1, number2, size):
    """Crea una matriz cuadrada size x size rellena con el valor number2 salvo en la esquina inferior izquierda donde hay una celda con el valor number1"""
    return matrix([[number1 if j == size and i==1 else number2 for i in (1..size)] for j in (1..size)])
    
def show_colored_box_matrix(m, field = None, big_blocks = False, subdivisions = True,
                       reverseChainOrder = False, reverseCompanionOrder = False, reverseOrder = None):
    """
    Dada una matriz pinta coloreada su matriz por bloques.
    
    Parámetros:
     - m es la matriz inicial
     - field es el cuerpo de trabajo, si no se da ninguno se toma el de la matriz
     - big_blocks (por defecto False) si lo ponemos a True toma la cadena toda unida como una única compañera de su cabeza
     - subdivisions (por defecto True) pone o no las líneas de corte en cada cadena
     - reverseChainOrder (por defecto False) toma como base la cadena de abajo a arriba
     - reverseCompanionOrder (por defecto False) toma como base en cada compeñera el orden normal o el orden inverso:  {fⁿ⁻¹(v), f^ⁿ⁻²(v), ..., f²(v), f(v), v}
     - reverseOrder (por defecto None) si se le da valor, este valor pisa a los dos anteriores parámetros
     
    Los colores elegidos son:
     - En rojo las matrices compañeras
     - En azul los enlaces de cadenas
     - En naraja la parte cero dentro de una cadena
     - En amarillo la parte cero dentro de cada invariante
     - En gris la parte cero fuera de cada invariante por invarianza
    """
    m = box_matrix(m, field, big_blocks, False, subdivisions, 
                   reverseChainOrder, reverseCompanionOrder, reverseOrder)
    dimen = length_of_chains(m, field)
    if reverseOrder is not None:
        reverseChainOrder = reverseOrder
        reverseCompanionOrder = reverseOrder
    
    color1 = 9
    color2 = 5
    color3 = 4
    color4 = 1
    color5 = 0
    
    matrixList = list()
    for key in dimen.keys():
        matrixList2 = list()
        deg = key.degree()
        for d in dimen[key]:
            # d = 5
            # deg = 2 
            matrixList3 = list()
            for i in (1..d):
                for j in (1..d):
                    if i==j or big_blocks:
                        matrixList3.append(constant_matrix(color5,deg))
                    elif i==j+1 and not reverseChainOrder and reverseCompanionOrder:
                        matrixList3.append(link_matrix_up(color4,color3, deg))
                    elif i==j+1 and not reverseChainOrder and not reverseCompanionOrder:
                        matrixList3.append(link_matrix_down(color4,color3, deg))
                    elif i+1==j and reverseChainOrder and reverseCompanionOrder:
                        matrixList3.append(link_matrix_up(color4,color3, deg))
                    elif i+1==j and reverseChainOrder and not reverseCompanionOrder:
                        matrixList3.append(link_matrix_down(color4,color3, deg))
                    else:
                        matrixList3.append(constant_matrix(color3, deg))
            matrixList2.append(block_matrix(matrixList3, nrows=d))
        matrixList2new = list()
        for i in range(len(matrixList2)):
            for j in range(len(matrixList2)):
                if i==j:
                    matrixList2new.append(matrixList2[i])
                else:
                    matrixList2new.append(constant_matrix(color2, matrixList2[i].nrows(), matrixList2[j].nrows()))
        matrixList.append(block_matrix(matrixList2new, nrows=len(matrixList2)))
    
    matrixListnew = list()
    for i in range(len(matrixList)):
        for j in range(len(matrixList)):
            if i==j:
                matrixListnew.append(matrixList[i])
            else:
                matrixListnew.append(constant_matrix(color1, matrixList[i].nrows(), matrixList[j].nrows()))
    aux = block_matrix(matrixListnew, nrows=len(matrixList))
    p = matrix_plot(aux, vmin=color5, vmax=color1, cmap='Pastel1', subdivisions=True, subdivision_style=dict(color='red',thickness=2))
    
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            p += text(str(m[i, j]), (j, i), color="black")
    show(p, frame=False)
    
def create_matrix(dimen, field=None):
    """
    Dado un diccionario que tiene por claves polinomios y por valores listas de enteros (representando la longitud de las distintas cadenas de bloques asocidadas a dicho polinomio) devuelve la matriz por bloques correspondiente. Útil en combinación con "randomize_base_in_matrix"
    
    Ejemplo:
    
    |   dic1 = {x^2-2:[1], x^2+1:[1], x+5: [3,2]}
    |   create_matrix(dic1)
    """
    if field is None:
        field = list(dimen.keys())[0].variables()[0].base_ring()
    matrixList = list()
    for key in dimen.keys():
        matrixList2 = list()
        deg = key.degree()
        for d in dimen[key]:
            # d = 5
            # deg = 2 
            matrixList3 = list()
            for i in (1..d):
                for j in (1..d):
                    if i==j:
                        matrixList3.append(matrix(field,companion_matrix(key)))
                    elif i==j+1:
                        matrixList3.append(matrix(field,link_matrix_down(1,0,deg)))
                    else:
                        matrixList3.append(matrix(field,constant_matrix(0,deg)))
            matrixList2.append(block_matrix(matrixList3, nrows=d))
        matrixList2new = list()
        for i in range(len(matrixList2)):
            for j in range(len(matrixList2)):
                if i==j:
                    matrixList2new.append(matrixList2[i])
                else:
                    matrixList2new.append(constant_matrix(0, matrixList2[i].nrows(), matrixList2[j].nrows()))
        matrixList.append(block_matrix(matrixList2new, nrows=len(matrixList2)))
    
    matrixListnew = list()
    for i in range(len(matrixList)):
        for j in range(len(matrixList)):
            if i==j:
                matrixListnew.append(matrixList[i])
            else:
                matrixListnew.append(constant_matrix(0, matrixList[i].nrows(), matrixList[j].nrows()))
    return block_matrix(matrixListnew, nrows=len(matrixList))

def randomize_base_in_matrix(m, field=None, random_level=1, forceField=False):
    """
    Dada una matriz cuadrada m busca una matriz semejante de forma aleatoria.
    
    Parámetros:
     - m: la matriz cuadrada inicial
     - field (opcional, si no se especifica se elige el anillo de la matriz)
     - random_level: 1 (poco) o 2 (mucho) indica el grado de mezcla de la nueva base. A mayor grado más enresevada en la matriz resultante
     - forceField (por defecto False). Indica si la matriz final debe estar en el mismo field que la inicial. Útil para forzar enteros.
    """
    if m.nrows() != m.ncols():
        raise ValueError("Input matrix is not square")
    if field is None:
        field = m.base_ring()
    size = m.nrows()
    rM = matrix(field,random_matrix(FiniteField(2),size,size,"unimodular"))
    for i in range(size):
        for j in range(size):
            if random_level == 1:
                rM[i,j] *= randrange(-1,2)*randrange(0,2)
            elif random_level == 2:
                rM[i,j] *= randrange(-1,2)
    if not rM.is_singular() and forceField:
        try:
            matrix(field,~rM)
        except TypeError:
            rM = zero_matrix(2)
    while rM.is_singular():
        rM = matrix(field,random_matrix(FiniteField(2),size,size,"unimodular"))
        for i in range(size):
            for j in range(size):
                if random_level == 1:
                    rM[i,j] *= randrange(-1,2)*randrange(0,2)
                elif random_level == 2:
                    rM[i,j] *= randrange(-1,2)
        if not rM.is_singular() and forceField:
            try:
                matrix(field,~rM)
            except TypeError:
                rM = zero_matrix(2)
    return rM*m*(~rM)    
    
def minimal_polynomial(A, v, var=x):
    """Calcula el polinomio mínimo asociado al vector coordenada v, usando la variable var, respecto al endomorfismo que tiene por matriz coordenada A"""
    V = v.parent()
    if v == V.zero():
        return 1
    S = minimal_invariant_subspace(A, v)
    coord = S.coordinates(A^S.dimension()*v)
    return var^len(coord) - annihilator_basis_to_expressions([coord],[var^i for i in range(len(coord))])[0]
    
def minimal_invariant_subspace(A, v):
    """Calcula el menor subespacio A-invariante que contiene al vector coordenada v"""
    V = v.parent()
    if v == V.zero():
        return span([v])
    S = V.subspace_with_basis([v])
    while(A*v not in S):
        v = A*v
        S = V.subspace_with_basis(S.basis() + [v])
    return S

# Práctica 2

def square_form(Q1, variables=None):
    """Dada una forma cuadrática muestra su expresión polinómica formando cuadrados
    
    Parámetros:
        - Q1: forma cuadrática
        - variables (por defecto (x0,x1,...)): lista de variables a usar
    """
    Q1 = QuadraticForm(QQ,Q1.dim(),Q1.coefficients())
    Q2, P2 = Q1.rational_diagonal_form(return_matrix=True)
    P = ~P2
    pol1 = Q1.polynomial() if variables is None else Q1.polynomial(names=variables)
    pol2 = Q2.polynomial() if variables is None else Q2.polynomial(names=variables)
    
    va = pol1.variables()
    v  = P * vector(va)
    
    vaO = list(map(lambda l: str(l), va))
    vaN = copy(vaO) #list(map(lambda s: s[0] + "n" + s[1:], vaO)) 
    
    chg = []
    for e in v:
        s1 = str(e)
        for i in range(len(vaO)):
            s1 = s1.replace(vaO[i],vaN[i])
        chg.append(s1)
        
    m = min([len(v) for v in vaO])
    chg = ["(" + v + ")" if len(v) > m + 1 else v for v in chg]
    
    repl = {vaO[i]: chg[i] for i in range(len(vaO))}
    
    pattern = '|'.join(map(re.escape, sorted(repl, key=len, reverse=True)))
    print("Before:",Q1.polynomial(names=vaO))
    return re.sub(pattern, lambda m: repl[m.group()], str(pol2))
    
def orthogonal_subspace(S, matrix):
    """Calcula el ortogonal de S respecto a una matriz
    
    Ejemplo:
    |   V = VectorSpace(QQ,3)
    |   S = V.subspace([[1,2,1],[0,2,1]])
    |   M = matrix(3,[0,0,0,0,0,0,0,0,1])
    |   orthogonal_subspace(S,M)
    """
    V = S.ambient_vector_space()
    F = S.base()
    anh = V.annihilator_basis(S.basis(),lambda x,y: vector(F,[x*matrix*y]))
    return V.subspace(anh)
    
def gram_schmidt(B,M,orthonormal=False):
    """ Dada una matriz fila o lista de vectores aplica Gram-Schmidt
    
    Parámetros:
        - B: matriz fila o lista de vectores que forman una base inicial
        - M: matriz de Gram del producto escalar
        - orthonormal (por defecto False): indica si se debe o no ortonormalizar la base
        
    Devuelve la matriz fila o lista ortogonalizada u ortonormalizada.
    
    Ejemplo:
    |   M = matrix([[2,1,0],[1,2,0],[0,0,1]])
    |   v = vector([1,2,1])
    |   w = vector([1,-1,0])
    |   gram_schmidt([v,w],M)
    """
    if B.__class__.__module__.startswith('sage.matrix.'):
        vlist = B.rows()
        lista = False
    elif isinstance(B, list | tuple):
        vlist = B
        lista = True
    else:
        raise ValueError("B no es matriz ni lista")
    nvlist = []
    for v in vlist:
        nvlist.append(v - sum([(v*M*w)/(w*M*w)*w for w in nvlist]))
    if(orthonormal):
        nvlist = [v/sqrt(v*M*v) for v in nvlist]
    if(lista):
        return nvlist
    else:
        return matrix(nvlist)
