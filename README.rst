=============================
mch-anemoi-plugins
=============================
Collection of [anemoi plugins](https://anemoi.readthedocs.io/projects/plugins/en/latest/index.html#) used at MeteoSwiss.

The **mch-anemoi** package bridges:

    - sources accessible from the **data-provider** package to *anemoi.sources*. As of now, sources that you can write in a yaml file are the following:

        - numerical weather prediction: ``cosmo1e``, ``icon1`` forecasts from **gridefix-read** package;
        - nowcasting: ``inca`` analysis from **gridefix-read** package;
        - topography: ``nasadem``, ``cedtm`` and ``dhm25`` using url requests from original sources;
        - observations: ``station`` data from **jretrieve api**;
        - satellite: ``satellite`` MSG variables from local cache;
        - radar: ``radar`` data from local cache.

    - base processing functions inspired/inherited from those available for xarray objects in the **gridefix-process** or **meteodatalab** package to *anemoi.filters*.

        - ``project`` to a target ``crs``;
        - ``interp2res`` to reach a target ``resolution``;
        - ``interp2grid`` to match a target ``template`` grid.

In the above description, text ``formatted like this`` are either sources, filters or arguments passed to filters that can be written in a yaml configuration file for creating a dataset using **anemoi-datasets**.

Variables available for each source listed above are listed `on the confluence page of the nowcasting data cache <https://meteoswiss.atlassian.net/wiki/spaces/Nowcasting/pages/322143175/Data+cache>`_. The variable name to write in the yaml file is the one written in the column *HL variable*.


The objective of this package is to use the **anemoi-datasets** package with *custom filters* and *internal mch sources*.
It should handle configurations looking like the file **anemoi_config.yaml**, even though this specific example should fail when we try to create the dataset with anemoi because station data and COSMO-1E data don't have the same time or spatial granularity. 