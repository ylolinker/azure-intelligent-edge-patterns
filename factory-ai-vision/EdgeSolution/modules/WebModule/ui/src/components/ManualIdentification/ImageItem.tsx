import React, { useState, useEffect, SetStateAction, Dispatch, FC, memo, useMemo, useCallback } from 'react';
import { Dropdown, DropdownItemProps, Text, Button, RadioGroup } from '@fluentui/react-northstar';
import LabelDisplayImage from '../LabelDisplayImage';
import LabelingPageDialog from '../LabelingPageDialog';
import { JudgedImageList, RelabelImage } from './types';

interface ImageIdentificationItemProps {
  confidenceLevel: number;
  relabelImages: RelabelImage[];
  imageIndex: number;
  setJudgedImageList: Dispatch<SetStateAction<JudgedImageList>>;
  partId: number;
  partItems: DropdownItemProps[];
  isPartCorrect: number;
}
const ImageIdentificationItem: FC<ImageIdentificationItemProps> = ({
  confidenceLevel,
  relabelImages,
  imageIndex,
  setJudgedImageList,
  partId,
  partItems,
  isPartCorrect,
}) => {
  const filteredPartItems = useMemo(
    () => [
      {
        header: 'No Object',
        content: {
          key: null,
        },
      },
      ...partItems.filter((e) => (e.content as any).key !== partId),
    ],
    [partId, partItems],
  );
  const [selectedPartItem, setSelectedPartItem] = useState<DropdownItemProps>(filteredPartItems[0]);

  const onDropdownChange = (_, { value }): void => {
    if (value !== null) {
      setSelectedPartItem(value);

      setJudgedImageList((prev) => {
        const next = [...prev];
        const idx = next.findIndex((e) => e.imageId === relabelImages[imageIndex].id);

        if (idx >= 0) next[idx] = { ...next[idx], partId: value.content.key };
        else {
          next.push({
            imageId: relabelImages[imageIndex].id,
            partId: value.content.key,
          });
        }

        return next;
      });
    }
  };

  const onRadioGroupChange = (_, newProps): void => {
    setJudgedImageList((prev) => {
      const next = [...prev];

      if (newProps.value === 1) {
        const idx = next.findIndex((e) => e.imageId === relabelImages[imageIndex].id);

        if (idx >= 0) next[idx] = { ...next[idx], partId };
        else next.push({ imageId: relabelImages[imageIndex].id, partId });
      }
      if (newProps.value === 0) {
        const idx = next.findIndex((e) => e.imageId === relabelImages[imageIndex].id);

        if (idx >= 0) next[idx] = { ...next[idx], partId: null };
        else {
          next.push({
            imageId: relabelImages[imageIndex].id,
            partId: null,
          });
        }
      }
      return next;
    });
  };

  useEffect(() => {
    setJudgedImageList([]);
    setSelectedPartItem(filteredPartItems[0]);
  }, [filteredPartItems, setSelectedPartItem, setJudgedImageList]);

  return (
    <div
      style={{
        display: 'flex',
        minHeight: '9em',
        maxHeight: '30%',
        justifyContent: 'center',
        padding: '1em',
      }}
    >
      <div style={{ margin: '0.1em' }}>
        <LabelDisplayImage labelImage={relabelImages[imageIndex]} width={100} height={100} />
      </div>
      <div
        style={{
          height: '100%',
          width: '40%',
          display: 'flex',
          flexFlow: 'column',
          justifyContent: 'space-between',
        }}
      >
        <Text truncated>
          Confidence Level: <b>{confidenceLevel}%</b>
        </Text>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            width: '100%',
            height: '50%',
          }}
        >
          <RadioGroup
            checkedValue={isPartCorrect}
            onCheckedValueChange={onRadioGroupChange}
            items={[
              {
                key: '1',
                label: 'Yes',
                value: 1,
              },
              {
                key: '0',
                label: 'No',
                value: 0,
              },
            ]}
          />
          {isPartCorrect === 0 && filteredPartItems.length > 0 && (
            <div style={{ width: '50%' }}>
              <Dropdown
                fluid
                items={filteredPartItems}
                onChange={onDropdownChange}
                value={selectedPartItem}
              />
            </div>
          )}
        </div>
        <LabelingPageDialog
          imageIndex={imageIndex}
          images={relabelImages}
          isRelabel={true}
          trigger={<Button primary content="Identify" disabled={!isPartCorrect} />}
        />
      </div>
    </div>
  );
};

export default memo(ImageIdentificationItem);