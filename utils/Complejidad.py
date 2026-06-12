def get_complexity(name):

    table = {
        "baseline": "Sin optimización",
        "sort": "O(n log n)",
        "heap": "O(n log k)",
        "quick": "O(n) promedio / O(n²) peor caso"
    }

    return table.get(name, "Desconocida")