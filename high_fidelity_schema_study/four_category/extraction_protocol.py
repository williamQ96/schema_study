"""Explicit routing for independently versioned grouped extraction contracts."""

GROUPED_PROTOCOLS = frozenset({'extraction-anchors/v3', 'extraction-surfaces/v4', 'extraction-pointers/v5', 'extraction-pointers/v6', 'extraction-local-quotes/v7', 'extraction-quote-options/v8', 'extraction-page-regions/v13', 'extraction-page-regions/v13r2'})


def module_for_task(task):
    identity = (task.get('schema_version'), task.get('admission_rules_version'))
    if identity == ('four-category-task/v13r2', 'extraction-page-regions/v13r2'):
        from . import extraction_v13r2

        return extraction_v13r2
    if identity == ('four-category-task/v13', 'extraction-page-regions/v13'):
        from . import extraction_v13
        return extraction_v13
    if identity == ('four-category-task/v5', 'extraction-anchors/v3'):
        from . import extraction_v5
        return extraction_v5
    if identity == ('four-category-task/v6', 'extraction-surfaces/v4'):
        from . import extraction_v6
        return extraction_v6
    if identity == ('four-category-task/v7', 'extraction-pointers/v5'):
        from . import extraction_v7
        return extraction_v7
    if identity == ('four-category-task/v8', 'extraction-pointers/v6'):
        from . import extraction_v8
        return extraction_v8
    if identity == ('four-category-task/v9', 'extraction-local-quotes/v7'):
        from . import extraction_v9
        return extraction_v9
    if identity == ('four-category-task/v10', 'extraction-quote-options/v8'):
        from . import extraction_v10
        return extraction_v10
    raise ValueError('unsupported_grouped_extraction_contract')
