import igraph as ig

if __name__ == "__main__":
	g = ig.Graph()
	print(g.add_vertex("start"))
	print(g.add_vertex("goal"))
	g.vs[0]["q"] = [1, 2, 3, 4, 5, 6, 7]
	g.vs[1]["q"] = [3, 2, 3, 2, 3, 2, 3]
	print(g.vs[0]["q"])
	print(g.vs["name"].index('start'))