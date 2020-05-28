
from factory_calc import FactoryCalculator

if __name__ == "__main__":
    blocks = [
        ({"electronic-circuit"}, {"iron-plate", "copper-plate"},),
        ({"advanced-circuit"}, {"electronic-circuit", "copper-plate", "plastic-bar"},),
        (
            {"processing-unit"},
            {"electronic-circuit", "advanced-circuit", "sulfuric-acid"},
        ),
    ]
    bus_item = {
        "copper-plate",
        "iron-plate",
        "stone-brick",
        "coal",
        "steel-plate",
        "water",
        "petroleum-gas",
        "electronic-circuit",
        "advanced-circuit",
        "processing-unit",
        "stone",
        "battery",
        "plastic-bar",
        "light-oil",
        "lubricant",
    }
    packs = {
        "automation-science-pack",
        "logistic-science-pack",
        "chemical-science-pack",
        "military-science-pack",
        "production-science-pack",
        "utility-science-pack",
        "space-science-pack",
    }
    blocks += (({p}, bus_item) for p in packs)

    factory = FactoryCalculator()
    factory.build_blocks(blocks)
    factory.solve({"7sp": 1.25})
    factory.print_result()