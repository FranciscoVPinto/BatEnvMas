#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 16:13:01 2025

@author: omega
"""


class DPTests:
    @classmethod
    def test_structure_equal(self,df1, df2):
        assert df1.shape == df2.shape, f"Shapes differ: {df1.shape} vs {df2.shape}"
        assert df1.columns.equals(df2.columns), f"Columns differ: {df1.columns} vs {df2.columns}"
        assert df1.index.equals(df2.index), f"Indices differ: {df1.index} vs {df2.index}"
        # print("DataFrame structure (shape, columns, index) is equal.")
