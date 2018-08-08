/**
 * Definition of SegmentTreeNode:
 * class SegmentTreeNode {
 * public:
 *     int l, r;
 *     SegmentTreeNode *left, *right;
 *     SegmentTreeNode(int l, int r) {
 *         this->l = l, this->r = r;
 *         this->left = this->right = NULL;
 *     }
 * }
 */

class SegmentTreeNode {
public:
    int l, r ;
    SegmentTreeNode *left, *right;
    SegmentTreeNode(int l, int r){
        this->l = l, this->r = r ;
        this->left = this->right = Null;
    }
}

class Solution {
public:
    /**
     *@param l, r: Denote an segment / interval
     *@return: The root of Segment Tree
     */
    SegmentTreeNode * building(int l, int r) {
        // write your code here
        SegmentTreeNode * node = new SegmentTreeNode(l, r);
        if (l == r)
            return node;
        int mid = (l + r) / 2;
        node->left = building(l, mid);
        node->right = building(mid+1, r);
        return node;
    }
};