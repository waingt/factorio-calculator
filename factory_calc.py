import ast
import json
import os

import numpy as np
from scipy.optimize import linprog


class assumption:
    slot_count = {
        "crafting": 4,
        "advanced-crafting": 4,
        "crafting-with-fluid": 4,
        "oil-processing": 3,
        "chemistry": 3,
        "smelting": 2,
        "centrifuging": 2,
        "rocket-building": 4,
        "lab": 2,
        "other": 0,
    }
    internal_speed = {
        "crafting": 1.25,
        "advanced-crafting": 1.25,
        "crafting-with-fluid": 1.25,
        "oil-processing": 1,
        "chemistry": 1,
        "smelting": 2,
        "centrifuging": 1,
        "rocket-building": 1,
        "lab": 3.5,  # if all research speed technologies has been unlocked
        "other": 1,
    }
    productivity_module_productivity_bouns = 0.1
    productivity_module_speed_bouns = -0.15
    speed_module_speed_bouns = 0.5
    extenal_speed_bouns_count = {  # the beacon amount,affecting a machine
        "crafting": 8,
        "advanced-crafting": 8,
        "crafting-with-fluid": 8,
        "oil-processing": 10,
        "chemistry": 22 / 3,
        "smelting": 8,
        "centrifuging": 8,
        "rocket-building": 8,
        "lab": 4,
        "other": 0,
    }


class ModulePolicy:
    def __init__(
        self,
        use_module: bool,
        use_speed_module_instead=True,
        extenal_speed_bouns=None,
        specification_on_recipes={},
        specification_on_category={},
    ):
        self.use_module = use_module
        self.use_speed_module_instead = use_speed_module_instead
        self.extenal_speed_bouns = (
            extenal_speed_bouns
            if extenal_speed_bouns
            else assumption.extenal_speed_bouns_count
        )
        self.specification_on_recipes = specification_on_recipes
        self.specification_on_category = {
            "block": {"multiplier": (1, 1)},
            "other": {"multiplier": (1, 1)},
        }
        self.specification_on_category.update(specification_on_category)

    def get_multiplier(self, recipe):
        "return production_multiplier and speed"
        recipe_name = recipe["name"]
        use_module = self.use_module
        category = recipe["category"]
        if category in self.specification_on_category:
            t = self.specification_on_category[category]
            if "multiplier" in t:
                return t["multiplier"]
            if "use_module" in t:
                use_module = t["use_module"]
            if "extenal_speed_bouns" in t:
                extenal_speed_bouns = t["extenal_speed_bouns"]
        if recipe_name in self.specification_on_recipes:
            t = self.specification_on_recipes[recipe_name]
            if "multiplier" in t:
                return t["multiplier"]
            if "use_module" in t:
                use_module = t["use_module"]
            if "extenal_speed_bouns" in t:
                extenal_speed_bouns = t["extenal_speed_bouns"]
        production_multiplier = 1
        speed = assumption.internal_speed[category]
        extenal_speed_bouns = self.extenal_speed_bouns[category]
        if use_module:
            slots = assumption.slot_count[category]
            if recipe["can_use_productivity_module"]:
                production_multiplier += (
                    assumption.productivity_module_productivity_bouns * slots
                )
                speed *= (
                    1
                    + extenal_speed_bouns
                    + assumption.productivity_module_speed_bouns * slots
                )
            else:
                speed *= (
                    1
                    + extenal_speed_bouns
                    + self.use_speed_module_instead
                    * slots
                    * assumption.speed_module_speed_bouns
                )
        return production_multiplier, speed


class RecipePolicy:
    def __init__(
        self,
        product_recipes_dict,
        *,
        given_recipes: set = None,
        prefered_recipes: dict = {},
        banned_recipes: set = None,
        dislike_recipes: set = {}
    ):
        self.product_recipes_dict = product_recipes_dict
        self.given_recipes = given_recipes
        if banned_recipes and type(banned_recipes) != set:
            banned_recipes = set(banned_recipes)
        self.banned_recipes = banned_recipes
        self.prefered_recipes = prefered_recipes

    def get(self, product):
        for r in self.prefered_recipes.values():
            if product in r["products"]:
                return [r]
        if product not in self.product_recipes_dict:
            return []
        if self.banned_recipes:
            return [
                r
                for r in self.product_recipes_dict[product]
                if r["name"] not in self.banned_recipes
            ]
        else:
            return self.product_recipes_dict[product]

    def add_prefered_recipe(self, recipe):
        self.prefered_recipes[recipe["name"]] = recipe


def read_recipe_file():
    def can_use_productivity_module(recipe: dict):
        if recipe["category"] == "smelting":
            return True
        group = recipe["group"]
        # "raw-material" "fluid-recipes" "intermediate-product" but for ["satellite","uranium-fuel-cell","nuclear-fuel-reprocessing"]
        if group in ["raw-material", "fluid-recipes", "science-pack"]:
            return True
        if group == "intermediate-product" and recipe["name"] not in [
            "satellite",
            "uranium-fuel-cell",
            "nuclear-fuel-reprocessing",
        ]:
            return True
        return False

    recipes_dict = {}
    product_recipes_dict = {}
    path = os.path.expandvars(r"%appdata%\Factorio\script-output\recipes")
    with open(path) as f:
        t = json.load(f)
    for i in t:
        new_products = {}
        new_ingredients = {}
        for j in i["products"]:
            amount = j["amount"]
            if "catalyst_amount" in j:
                amount -= j["catalyst_amount"]
                if amount <= 0:
                    continue
            amount *= j.get("probability", 1)
            new_products[j["name"]] = amount
        for j in i["ingredients"]:
            amount = j["amount"]
            if "catalyst_amount" in j:
                amount -= j["catalyst_amount"]
                if amount <= 0:
                    continue
            new_ingredients[j["name"]] = amount
        i["products"] = new_products
        i["ingredients"] = new_ingredients
        i["can_use_productivity_module"] = can_use_productivity_module(i)
        recipes_dict[i["name"]] = i
    recipes_dict.update(
        {
            "space-science-pack": {
                "name": "space-science-pack",
                "category": "other",
                "energy": 42.5,
                "ingredients": {"rocket-part": 100, "satellite": 1},
                "products": {"space-science-pack": 1000},
                "can_use_productivity_module": False,
            },
            "7sp": {
                "name": "7sp",
                "category": "lab",
                "energy": 60,
                "ingredients": {
                    "automation-science-pack": 1,
                    "logistic-science-pack": 1,
                    "chemical-science-pack": 1,
                    "production-science-pack": 1,
                    "military-science-pack": 1,
                    "utility-science-pack": 1,
                    "space-science-pack": 1,
                },
                "products": {"7sp": 1},
                "can_use_productivity_module": False,
            },
        }
    )
    for i in recipes_dict.values():
        for j in i["products"].keys():
            if j not in product_recipes_dict:
                product_recipes_dict[j] = [i]
            else:
                product_recipes_dict[j].append(i)
    return product_recipes_dict


def search_path(
    recipe_policy: RecipePolicy, products: dict, ingredients: set,
):
    items = set(products)
    intermediates = set()
    recipes = {}
    ingredients_ = {}
    count = 0
    if recipe_policy.given_recipes:
        recipes = dict.fromkeys(recipe_policy.given_recipes)
    else:
        while len(items) > 0:
            item = items.pop()
            t = recipe_policy.get(item)
            if len(t) == 0:
                if ingredients == set():
                    ingredients_[item] = None
                else:
                    assert False, "归结不到原材料"
            for recipe in t:
                for k in recipe["ingredients"]:
                    if k in ingredients:
                        ingredients_[k] = None
                    else:
                        items.add(k)
                        intermediates.add(k)
                recipes[recipe["name"]] = recipe
                count += 1
                assert count < 1e5, "搜索次数过多，可能存在环路"
    intermediates -= products.keys()
    intermediates -= ingredients_.keys()
    return recipes, intermediates, ingredients_


def solve(
    module_policy: ModulePolicy,
    recipe_policy: RecipePolicy,
    products: dict,
    ingredients: set = set(),
):
    "return required factory for each recipe and amount of ingredients"
    recipes, intermediates, ingredients_ = search_path(
        recipe_policy, products, ingredients
    )
    name_position_dict = {}
    count = 0
    b = np.zeros(len(products) + len(intermediates))
    for k, v in products.items():
        name_position_dict[k] = count
        b[count] = v
        count += 1
    for k in intermediates:
        name_position_dict[k] = count
        count += 1
    count = 0
    for k in ingredients_:
        name_position_dict[k] = count
        count += 1
    row, col = len(products) + len(intermediates), len(recipes)
    A = np.zeros((row, col))
    c = np.zeros(col)
    d = np.zeros((len(ingredients_), col))
    for i, name in enumerate(recipes):
        r = recipes[name]
        m, speed = module_policy.get_multiplier(r)
        c[i] = r["energy"] / speed
        for k, v in r["products"].items():
            A[name_position_dict[k], i] = v * m
        for k, v in r["ingredients"].items():
            if k not in ingredients_:
                A[name_position_dict[k], i] = -v
            else:
                d[name_position_dict[k], i] = v
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        assert False, "可能有多重途径，无法解决"
    c = c * x
    d = d @ x
    factories = {name: c[i] for i, name in enumerate(recipes)}
    for i, k in enumerate(ingredients_):
        ingredients_[k] = d[i]
    return factories, ingredients_


def solve_linprog(
    specification: str,
    module_policy: ModulePolicy,
    recipe_policy: RecipePolicy,
    products: dict,
    ingredients: set = set(),
):
    recipes, intermediates, ingredients_ = search_path(
        recipe_policy, products, ingredients
    )
    name_position_dict = {}
    recipe_position_dict = {}
    count = 0
    b_eq = np.zeros(len(intermediates))
    for k in products.keys():
        name_position_dict[k] = count
        count += 1
    for k in intermediates:
        name_position_dict[k] = count
        count += 1
    count = 0
    for k in ingredients_:
        name_position_dict[k] = count
        count += 1
    row, col = len(products) + len(intermediates), len(recipes)
    A = np.zeros((row, col))
    e = np.zeros(col)
    g = np.zeros((len(ingredients_), col))
    for i, name in enumerate(recipes):
        recipe_position_dict[name] = i
        r = recipes[name]
        m, speed = module_policy.get_multiplier(r)
        e[i] = r["energy"] / speed
        for k, v in r["products"].items():
            A[name_position_dict[k], i] = v * m
        for k, v in r["ingredients"].items():
            if k not in ingredients_:
                A[name_position_dict[k], i] = -v
            else:
                g[name_position_dict[k], i] = v
    A_eq = A[len(products) :, :]
    p = A[: len(products), :]
    c = np.zeros(col)
    lines = specification.split("\n")
    goal, name = lines[0].split(" ")
    assert goal in ("min", "max")
    t = 1 if goal == "min" else -1
    if name[0] == "#":
        name = name[1:]
        c[recipe_position_dict[name]] = t
    else:
        c = A[name_position_dict[name], :] * t
    A_ub = np.zeros((len(lines) - 1, col))
    b_ub = np.zeros(len(lines) - 1)
    for i, l in enumerate(lines[1:]):
        name, comparsion, amount = l.split(" ")
        comparsion = 1 if comparsion in ("<", "<=") else -1
        amount = float(eval(amount))
        b_ub[i] = comparsion * amount
        if name[0] == "#":
            name = name[1:]
            A_ub[i, recipe_position_dict[name]] = comparsion
        elif name in products:
            A_ub[i, :] = comparsion * A[name_position_dict[name], :]
        elif name in intermediates:
            A_ub[i, :] = np.clip(comparsion * A[name_position_dict[name], :], 0, None)
        elif name in ingredients_:
            A_ub[i, :] = comparsion * g[name_position_dict[name], :]
        else:
            assert False, "无法识别名字"
    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    assert res.status == 0
    x = res.x
    e = e * x
    g = g @ x
    p = p @ x
    factories = {name: e[i] for i, name in enumerate(recipes)}
    for i, k in enumerate(ingredients_):
        ingredients_[k] = g[i]
    for i, k in enumerate(products):
        products[k] = p[i]
    return factories, ingredients_, products


def build_block(
    module_policy: ModulePolicy,
    recipe_policy: RecipePolicy,
    products: dict,
    ingredients: set,
):
    if type(products) == set:
        products = dict.fromkeys(products, 1)
    block_name = "Block_" + "+".join(products)
    factories, ingredients = solve(module_policy, recipe_policy, products, ingredients,)
    recipe = {
        "name": block_name,
        "products": products,
        "ingredients": ingredients,
        "energy": 1,
        "factories": factories,
        "category": "block",
    }
    recipe_policy.add_prefered_recipe(recipe)


class FactoryCalculator:
    def __init__(self, module_policy=ModulePolicy(False), recipe_policy=None):
        self.product_recipes_dict = read_recipe_file()
        self.result_dict = {}
        self.module_policy = module_policy
        if recipe_policy == None:
            self.recipe_policy = RecipePolicy(
                self.product_recipes_dict,
                banned_recipes=(
                    "basic-oil-processing",
                    "coal-liquefaction",
                    "solid-fuel-from-heavy-oil",
                    "solid-fuel-from-petroleum-gas",
                ),
            )
        else:
            self.recipe_policy = recipe_policy

    def reset(self):
        self.result_dict = {}
        self.module_policy = ModulePolicy(False)
        self.recipe_policy = RecipePolicy(
            self.product_recipes_dict,
            banned_recipes=(
                "basic-oil-processing",
                "coal-liquefaction",
                "solid-fuel-from-heavy-oil",
                "solid-fuel-from-petroleum-gas",
            ),
        )

    def build_block(self, products: dict, ingredients: set, module_policy=None):
        build_block(
            module_policy if module_policy else self.module_policy,
            self.recipe_policy,
            products,
            ingredients,
        )

    def build_blocks(self, blocks: list):
        for b in blocks:
            self.build_block(*b)

    def solve(
        self, products: dict, ingredients: set = set(),
    ):
        t = solve(self.module_policy, self.recipe_policy, products, ingredients)
        t = {["factories", "ingredients"][i]: t[i] for i in range(len(t))}
        self.result_dict.update(t)

    def solve_linprog(
        self, specification: str, products: dict, ingredients: set = set(),
    ):
        t = solve_linprog(
            specification, self.module_policy, self.recipe_policy, products, ingredients
        )
        t = {["factories", "ingredients", "products"][i]: t[i] for i in range(len(t))}
        self.result_dict.update(t)

    def print_result(self):
        def print_dict(d: dict):
            l = max((len(k) for k in d))
            f = "    {0:<%d} : {1:>6.2f}" % l
            for k in sorted(d):
                print(f.format(k, d[k]))

        if "factories" in self.result_dict:
            factories = self.result_dict["factories"]
            print()
            print("factories:")
            print_dict(factories)

        if "ingredients" in self.result_dict:
            ingredients = self.result_dict["ingredients"]
            print()
            print("ingredients:")
            print_dict(ingredients)

        if "products" in self.result_dict:
            products = self.result_dict["products"]
            print()
            print("products:")
            print_dict(products)

        if "factories" in self.result_dict:
            for k in sorted(factories):
                if k in self.recipe_policy.prefered_recipes:
                    amount = factories[k]
                    recipe = self.recipe_policy.prefered_recipes[k]
                    block_factories = recipe["factories"]
                    block_factories = {
                        k: v * amount for k, v in block_factories.items()
                    }
                    ingredients = recipe["ingredients"]
                    ingredients = {k: v * amount for k, v in ingredients.items()}
                    products = recipe["products"]
                    products = {k: v * amount for k, v in products.items()}
                    print()
                    print(recipe["name"])
                    print("    ", ingredients, "=>", products)
                    print_dict(block_factories)
