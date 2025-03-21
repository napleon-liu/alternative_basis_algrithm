#include <iostream>
#include <bits/stdc++.h>
#include <vector>
using namespace std;
using Matrix = vector<vector<double>>;
// 基变换矩阵
// const Matrix psi_opt = {
//     {1, 0, 0, 0},
//     {0, 1, -1, 1},
//     {0, 0, -1, 1},
//     {0, 1, 0, 1}};

// // 基变换逆矩阵
// const Matrix psi_opt_inv = {
//     {1, 0, 0, 0},
//     {0, 1, -1, 0},
//     {0, -1, 0, 1},
//     {0, -1, 1, 1}};

const Matrix U = {
    {0, 0, 0, 1},
    {0, 0, 1, 0},
    {0, 1, 0, 0},
    {1, 0, 0, 0},
    {0, 1, -1, 0},
    {-1, 1, 0, 0},
    {0, -1, 0, 1}};

const Matrix V = {
    {0, 0, 0, 1},
    {0, 0, 1, 0},
    {0, 1, 0, 0},
    {1, 0, 0, 0},
    {0, -1, 0, 1},
    {0, 1, -1, 0},
    {-1, 1, 0, 0}};

const Matrix W = {
    {0, 0, 0, 1},
    {0, 0, 1, 0},
    {0, 1, 0, 0},
    {1, 0, 0, 0},
    {1, 1, 0, 0},
    {0, -1, 0, -1},
    {0, 1, 1, 0}};