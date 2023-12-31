CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230823000000_e20230823235959_p20230824021715_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-08-24T02:17:15.793Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-08-23T00:00:00.000Z   time_coverage_end         2023-08-23T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��   �          @o\)���H�S33>��@\)Cr�{���H�K�?h��Ab�RCq�R                                    Bx���  �          @dz��ff�C�
�������
Coz��ff�C�
>��R@���CoxR                                    Bx��+L  �          @c33��Q��L�;���CuǮ��Q��J=q?�\A{Cu�                                    Bx��9�  �          @c33�ٙ��0�׿�Q���p�CnaH�ٙ��<(��(��$(�Cp                                      Bx��H�  �          @hQ���R�&ff�\(��aG�CeY���R�.{���
��
=Cf��                                    Bx��W>  �          @dz�����R��p���p�C_������R��G�����CbǮ                                    Bx��e�  �          @a��
=q�  ��(����HCbG��
=q�#33��p����
Ce�{                                    Bx��t�  �          @Z�H�'
=�(��ff�(�CA��'
=���\��
=�
ffCIJ=                                    Bx���0  �          @Tz��(�ÿ&ff��=q��CA���(�ÿ�  ��z���CH��                                    Bx����  �          @XQ��)����=q��\)���CN���)���У׿������CS��                                    Bx���|  �          @j=q�3�
����޸R��ffCP���3�
��{��33��=qCUs3                                    Bx���"  
�          @c�
�+���=q�����
=CR�=�+���{��Q����\CV                                    Bx����  �          @vff�7
=�ff����~�RC[xR�7
=�!G��\)�p�C]aH                                    Bx���n  �          @w
=�2�\��������
C[ٚ�2�\�"�\�J=q�<��C^Q�                                    Bx���  �          @{��5��ÿ�ff���HC\{�5�&ff�J=q�9�C^�                                    Bx���  �          @|(��AG��G���\)��ffCY��AG���Ϳ#�
���C[&f                                    Bx���`  �          @y���AG��p������~ffCX.�AG��Q������CZ:�                                    Bx��  �          @Z�H�,(����H���R��  CTY��,(����^�R�j�HCW�                                    Bx���  �          @p  �8Q����fff�_�
CY!H�8Q��z��G���  CZ�{                                    Bx��$R  �          @Tz��%��Q���Ϳ���CX��%��>�z�@�{CX�                                     Bx��2�  �          @:�H�녿�p�>�A�HCY��녿˅?Q�A�
=CV�                                    Bx��A�  �          @.{�z��G��\)�B�\C\Y��z��  >aG�@�33C\:�                                    Bx��PD  T          @-p��
=q�У�=�\)?�CY\�
=q��=q>���A�
CX:�                                    Bx��^�  �          @���녿�  �aG����CS��녿��\=#�
?k�CS��                                    Bx��m�  �          @6ff�\)����\�%G�CX���\)�޸R�#�
�J=qCY�q                                    Bx��|6  �          @c�
�0����
�Tz��Y��CX�{�0���(��Ǯ�ʏ\CZ}q                                    Bx����  �          @b�\���Q�E��LQ�C`ff���\)��=q��33Ca��                                    Bx����  �          @g
=�"�\�33���\��z�C^(��"�\�p�����\C`
                                    Bx���(  �          @hQ��1녿�Q쿣�
��p�CV�q�1��
�H�Y���YG�CY��                                    Bx����  �          @g��.�R�   ��p���p�CX=q�.�R�p��J=q�J{C[�                                    Bx���t  �          @[��'
=��녿ٙ���=qCP\�'
=���H��\)��G�CUB�                                    Bx���  �          @dz��/\)������
=�(�CK���/\)���ÿ����33CQ�                                    Bx����  �          @\)��녿��\��G���\)Cx33��녿��\=���@�z�Cx5�                                    Bx���f  �          @G
=@{��Q�?�\B
=C���@{<��
?�ffBQ�>��                                    Bx��   T          @XQ�@33��{@�B!�
C��@33�!G�@
=B2�HC�K�                                    Bx���  �          @\(�@	����G�@p�B 
=C�k�@	�����\@{B8p�C�L�                                    Bx��X  �          @P  @Q쿡G�@ffB!G�C��H@Q�L��@z�B6Q�C���                                    Bx��+�  �          @J�H?�p����?��A�
=C��R?�p���
=?��
B	C��\                                    Bx��:�  �          @A�?O\)�:=q������\)C�˅?O\)�:=q>�z�@��C���                                    Bx��IJ  �          @'
=>���ÿ��\��G�C�Y�>��
=�L�����\C��f                                    Bx��W�  �          @A�<��+�@5�B�33C��<��u@:=qB�B�C�                                      Bx��f�  �          @?\)��Ϳu@{B���Cp(���;�
=@'�B�G�CYW
                                    Bx��u<  �          @(Q������H?�{B6(�Cw������?��Bb(�CqB�                                    Bx����  �          @.{=u��>\A�C���=u�33?aG�A��C���                                    Bx����  �          @5����H�ff>�z�@���Cp����H���R?:�HA�G�CnǮ                                    Bx���.  T          @>�R���ÿ�Q쿣�
���
Cb�׿����
�H�W
=���\Cf
=                                    Bx����  ^          @dz���
�'���R��Cd�
��
�,(��L�Ϳ:�HCeY�                                   Bx���z  T          @r�\�(Q��+����
���Cas3�(Q��+�>��@|(�Ca�                                    Bx���   T          @y���\)�:=q�(���  CeaH�\)�>�R���
��Q�Cf
                                    Bx����  �          @y�����/\)������\)Cd}q���>{�:�H�,z�Cf��                                    Bx���l  �          @z�H�  �'������\)CeL��  �<(���Q����RCh��                                    Bx���  "          @_\)����33�p��  Cn𤿱��.{����Cs�                                    Bx���  
6          @���<���z������HCZ5��<���'���{�z�RC]��                                    Bx��^  ,          @����0������
����CY�=�0��� �׿Ǯ��
=C^Q�                                    Bx��%  �          @��\�/\)��=q����G�CU���/\)�33�������HC[��                                    Bx��3�  �          @�G��0  �z��ff��\)CX�q�0  ��R��{���C]�q                                    Bx��BP  �          @~{�:�H���H�p����CN�=�:�H��33�����ߙ�CU{                                    Bx��P�  �          @e�'�>#�
�=q�*p�C0n�'���Q�����(�\C;ٚ                                    Bx��_�  �          @e��1��ff�0���4(�CY
�1���;aG��fffCZaH                                    Bx��nB  �          @B�\��(���?Tz�A��Cq޸��(��33?�ffA��Co8R                                    Bx��|�  �          @]p�?u��(�@<��Bl�
C���?u�L��@L��B�W
C�4{                                    Bx����  �          @_\)?�=q��33@,��BK�C���?�=q���\@C33Bu=qC�=q                                    Bx���4  �          @_\)?�  ���@z�B'�C�S3?�  ��=q@2�\BT��C�O\                                    Bx����  �          @Vff?�ff��(�@�RB?��C�f?�ff���@6ffBj33C��
                                    Bx����  �          @\��>���=p�?�Q�A�p�C�w
>���!G�@�\B){C��R                                    Bx���&  �          @e�Ǯ�J=q?�33A��C�~��Ǯ�.{@33B {C��\                                    Bx����  �          @c�
�L���J�H?���A��HC���L���/\)@{B
=C��H                                    Bx���r  �          @Y������I��>�G�@�p�C�@ ����>�R?�\)A��C��                                    Bx���  �          @\(�>���S�
?=p�AI��C�G�>���C�
?���A�Q�C�b�                                    Bx�� �  �          @\(���33�I���   ���Cy𤿓33�L(�>L��@Z�HCz#�                                    Bx��d  �          @h�ÿxQ��Z=q������33C~
=�xQ��Y��>�(�@�p�C}�q                                    Bx��
  �          @Y�����
�K�=���?�ffC|Ϳ��
�E�?O\)A^�HC{�                                     Bx��,�  �          @XQ쿽p��@�׾aG��p��Cs�{��p��?\)>�ff@�
=Cs��                                    Bx��;V  
�          @Z�H�s33�L(��k��y��C}^��s33�J=q>��HA=qC}8R                                    Bx��I�  �          @j�H�����[�>#�
@$z�Cz�Ῑ���S�
?k�Aj�HCz�                                    Bx��X�  �          @fff��p��R�\>��H@�\)Cy�Ϳ�p��Fff?��HA�  Cxc�                                    Bx��gH  �          @_\)��\)�C�
?���A�p�C�\)��\)�/\)?�  B
=C�                                    Bx��u�  �          @c33>Ǯ�L(�?��HAÙ�C�u�>Ǯ�2�\@��B
=C���                                    Bx����  �          @`  ?��Dz�?�G�AΣ�C��?��*=q@
=qB33C��\                                    Bx���:  �          @k�@33����@�RB)\)C��=@33�Y��@/\)B@�C�Ф                                    Bx����  �          @n{@����@��B!�\C��@�}p�@,(�B:\)C��3                                    Bx����  �          @l��@	���޸R@ffB!��C���@	����z�@+�B>�RC��f                                    Bx���,  �          @tz�@(Q�˅@\)BG�C�z�@(Q쿅�@"�\B'�C�<)                                    Bx����  �          @z�H@(���\@\)Bz�C�@(���z�@5�B9G�C�Q�                                    Bx���x  �          @\)@
�H��@%B#(�C��@
�H��Q�@?\)BD�C�8R                                    Bx���  �          @�  @!녿�G�@#33B��C��R@!녿���@8��B8��C��                                    Bx����  �          @�G�@!녿�=q@!�B�C�
=@!녿��H@8��B733C�:�                                    Bx� j  �          @x��@�Ϳ���@"�\B$G�C���@�Ϳu@5�B<ffC�Y�                                    Bx�   �          @xQ�@"�\���\@(Q�B+�C��
@"�\�!G�@6ffB>
=C���                                    Bx� %�  �          @y��@1녿�\)@�RB��C��@1녿�@*�HB-��C��                                    Bx� 4\  �          @q�@*=q�h��@!G�B'�C���@*=q���R@*�HB3�\C���                                    Bx� C  �          @xQ�@'
=�c�
@.{B2=qC��3@'
=�u@6ffB=z�C�T{                                    Bx� Q�  �          @w
=@Q쿃�
@5BH�
C�{@Q쾳33@@  BY(�C�S3                                    Bx� `N  �          @u?Q���R@.�RB8�HC�  ?Q녿��@N�RBj�\C�L�                                    Bx� n�  �          @w����2�\@*=qB.z�C������ff@N�RBc�
C���                                    Bx� }�  �          @u�?z��!G�@3�
B=�C�n?z��ff@Tz�BqQ�C���                                    Bx� �@  �          @tz�>�p��!�@3�
B>�C�#�>�p���@Tz�BsQ�C���                                    Bx� ��  T          @q�>�  �'�@,��B7  C��
>�  ��@O\)Bl\)C���                                    Bx� ��  �          @j=q>���@.{B@p�C��\>��޸R@Mp�Bv�C�0�                                    Bx� �2  �          @mp�>��p�@-p�B<�C���>���G�@Mp�Bqz�C��)                                    Bx� ��  �          @p��?�녿У�@=p�BP{C���?�녿c�
@P��Bp��C��{                                    Bx� �~  �          @c�
?�  ��z�@:=qBZz�C�b�?�  �0��@J=qByG�C��)                                    Bx� �$  �          @G�>�33�"�\?У�B�HC��>�33�@�B7�C��
                                    Bx� ��  �          @>{�W
=�:=q>Ǯ@�C��W
=�/\)?��A���C��\                                    Bx�p  �          @E����333?�
=A��C��H����(�?�B�\C�J=                                    Bx�  �          @8�þ��
�0  ?@  Ar�\C��쾣�
�   ?���A噚C�]q                                    Bx��  �          @E���8��?k�A�G�C�w
���%?�=qA�ffC��
                                    Bx�-b  �          @B�\�(��1�?�=qA�{C��=�(��(�?ٙ�BffC�3                                    Bx�<  �          @@�׿��0  ?��A��\C�� ���=q?�(�B
=C��H                                    Bx�J�  �          @C�
����0  ?�p�A��C�*=����Q�?���B{C�n                                    Bx�YT  �          @H�ÿY���,��?�{A��C|}q�Y���33?�(�B(�Cy�f                                    Bx�g�  �          @R�\���:=q?�A͙�C�⏿��\)@�B
=C��                                    Bx�v�  �          @S33�!G��:�H?�33A��HC��{�!G��   @�
B\)C�3                                    Bx��F  �          @S�
�(���>{?��A�Q�C��q�(���$z�?�(�B(�C�H                                    Bx���  �          @S33�E��1�?ǮA�z�C~��E���@�B&z�C{�                                    Bx���  �          @J=q���1�?���A��C�*=���Q�?��HBQ�C�8R                                    Bx��8  �          @Fff�z��5?�ffA��
C�Ff�z��   ?��HBz�C���                                    Bx���  �          @C�
��Q��8Q�?xQ�A��RC�l;�Q��#33?��B��C���                                    Bx�΄  �          @AG���  �5?xQ�A�(�C��׾�  �!G�?У�B�HC�4{                                    Bx��*  �          @\(��\�3�
?�  A�=qC�!H�\�33@Q�B6�C�H�                                    Bx���  
}          @^{����S33?fffApz�C�Ff����>�R?�A��C�Ǯ                                    Bx��v  �          @2�\�#�
�  ?�=qB��C��ͽ#�
��ff@�BD\)C�o\                                    Bx�	  
�          @E�>����{?�\B�C��H>������H@z�BE�C�ff                                    Bx��  
�          @L��>�z��>{?���A���C��>�z��'
=?��B	(�C�&f                                    Bx�&h  "          @#33?���\)@p�B}�C�l�?��>��@
�HBw�A��
                                    Bx�5  �          @r�\?�����G�@[�B��C�޸?���?=p�@VffB�B�A�                                      Bx�C�  �          @~{?�p���Q�@W�Bl��C���?�p�>�@W
=Bk33AY�                                    Bx�RZ  �          @�G�@�ÿ:�H@p��Ba��C��@��>aG�@tz�Bgz�@�33                                    Bx�a   �          @��H@�ÿ
=q@w�Bfp�C���@��>�(�@xQ�Bg�RA#�
                                    Bx�o�  �          @��R@�Ϳ0��@s�
Bk{C�>�@��>�=q@w
=Bp�@�                                      Bx�~L  �          @���@�\)@tz�Bg33C�O\@>��@uBh��A�\                                    Bx���  �          @���@����@s33Bd��C���@��>�G�@s�
Be�A'�
                                    Bx���  T          @�G�@��&ff@u�Bg{C�@ @�>���@w�Bj�A ��                                    Bx��>  �          @���@�R�z�@q�B`33C�n@�R>\@s33Bbp�A��                                    Bx���  �          @�33@���0��@c�
B[��C��\@��>aG�@g�Ba��@�                                      Bx�Ǌ  
�          @�G�@���33@&ffB#z�C���@�Ϳ�=q@B�\BF�
C�g�                                    Bx��0  �          @���@�\�\)@=qB�HC�˅@�\�Ǯ@9��B9{C��{                                    Bx���  	�          @��@{�@ ��BQ�C��H@{���@=p�B8�C�P�                                    Bx��|  
�          @�  @=q��ff@HQ�B>(�C��
@=q�5@Z�HBV�HC�                                    Bx�"  
�          @�p�@(�ÿ��
@R�\BA  C��@(�þ��@`��BR�RC��                                     Bx��  
Z          @���@4z῱�@P��B8ffC��=@4z��\@`��BK
=C��R                                    Bx�n  
�          @�@<�Ϳ�G�@\��B?��C��@<�ͽ���@fffBJz�C���                                    Bx�.  �          @�G�@3�
��  @o\)BM��C�1�@3�
���
@w�BX33C��R                                    Bx�<�  �          @�(�@>�R��=q@mp�BEC�@>�R����@w
=BQ33C�f                                    Bx�K`  
�          @�33@AG���ff@i��BC(�C�b�@AG����
@s33BN
=C�1�                                    Bx�Z  
�          @�\)@?\)���@Z�HB9\)C�4{@?\)�Ǯ@h��BIp�C�Q�                                    Bx�h�  "          @�z�@@�׿�Q�@Mp�B/C�7
@@�׿�@^{BBz�C���                                    Bx�wR  T          @�  @7���ff@Tz�B1ffC��
@7��c�
@j�HBJ�\C�k�                                    Bx���  
�          @���@6ff���R@S�
B.\)C���@6ff����@mp�BJ�\C��\                                    Bx���  
�          @�=q@E���(�@HQ�B"
=C��)@E����@a�B<��C�=q                                    Bx��D  
�          @���@fff��(�@5�BffC�7
@fff�n{@K�B"(�C��H                                    Bx���  "          @�33@dz��\@1�B��C��3@dz�z�H@H��B!p�C�G�                                    Bx���  "          @�33@]p��@+�B{C�w
@]p�����@HQ�B �C��{                                    Bx��6  T          @��
@Q��\)@4z�Bz�C��R@Q녿�@S33B*��C�@                                     Bx���  "          @�=q@U��@0  B�
C�f@U���@L��B'�C�l�                                    Bx��  T          @�ff@R�\��@(�A��\C���@R�\�Ǯ@<��B�C�J=                                    Bx��(  �          @�\)@@  ���H@Q�A��C�q�@@  ��=q@$z�B=qC��                                    Bx�	�  �          @�G�@^{��  ?���A�(�C��@^{�J=q?��A޸RC���                                    Bx�t  �          @|(�>���s33?333A&�\C��>���^�R?�
=Ạ�C�b�                                    Bx�'  
�          @mp���\)�`  >�G�@��HC|.��\)�P��?���A���Cz�3                                    Bx�5�  �          @r�\��ff�e>��@���C}�q��ff�X��?���A��
C|�
                                    Bx�Df  
Z          @q녾�Q��n{>�(�@�Q�C�5þ�Q��^�R?�33A�ffC��                                    Bx�S  �          @qG����
�k�?Q�AH(�C��3���
�U�?�G�A�\)C��                                    Bx�a�  T          @n{�\�k�>�33@�
=C�\�\�\��?���A��\C��q                                    Bx�pX  	�          @i���\(��`  >��@�C�3�\(��P  ?��A��
C)                                    Bx�~�  
�          @b�\��  �W�>��R@�
=C}���  �J=q?�Q�A�z�C|�                                    Bx���  �          @h�ÿO\)�aG�>u@vffC���O\)�U�?�
=A��
C�'�                                    Bx��J  1          @p  �aG��e����}p�C�  �aG��a�?8Q�A2�RC�H                                    Bx���  
K          @{���ff�x�þk��Z�HC��R��ff�s�
?Q�AAp�C��f                                    Bx���  �          @mp��aG��b�\������\)C���aG��aG�?�A\)C�                                    Bx��<  �          @~�R��(��a�?}p�Ag�Cwh���(��H��?�z�A��Ct�H                                    Bx���  
�          @y����G��e�?��\Au��C~Q쿁G��J�H?���A�\)C|h�                                    Bx��  �          @b�\�^�R�Z=q>�\)@���C���^�R�Mp�?�Q�A�{C~��                                    Bx��.  �          @b�\�Y���[�>L��@Q�C�H�Y���P  ?���A��\C@                                     Bx��  �          @b�\�O\)�X�ý#�
�(��C�>��O\)�QG�?c�
Ak\)C��                                    Bx�z  �          @]p����\�N�R�#�
�*�HC|�����\�Q�>�=q@�
=C|Ǯ                                    Bx�    T          @\(��:�H�Fff��G����HC�T{�:�H�TzᾸQ��ÅC��                                    Bx�.�  T          @^�R�+��HQ쿘Q����HC���+��U���\)��Q�C�W
                                    Bx�=l  T          @u�W
=�e�?��A�=qC�L;W
=�E�@\)B��C�f                                    Bx�L  �          @�Q쾀  �g
=?޸RA���C����  �?\)@*=qB&G�C���                                    Bx�Z�  �          @s33��{�Fff?��A�{CzT{��{�p�@*=qB2��Cu�q                                    Bx�i^  T          @z�H�^�R�g
=?���A�G�C�<)�^�R�H��@	��B�C~�                                    Bx�x  "          @p�׿:�H�i��>��H@���C�S3�:�H�W�?��RA�ffC�޸                                    Bx���  T          @X�þ���Vff>�{@��\C�������HQ�?�G�A�\)C��{                                    Bx��P  �          @n�R���g
=?.{A+33C�ý��Q�?�z�A���C��q                                    Bx���  �          @��׽����u�?�Q�A���C�C׽����Vff@p�B��C�(�                                    Bx���  �          @��H��G��|(�?uAZffC����G��`��@�A�=qC�k�                                    Bx��B  T          @��H���\)?
=AffC�����j=q?�Q�Aď\C�g�                                    Bx���  T          @�G���ff�~{>\@�
=C���ff�mp�?�p�A�p�C��f                                    Bx�ގ  T          @~�R��G��{��u�\(�C�˅��G��r�\?�ffAv=qC���                                    Bx��4  
�          @y���Q��qG��k��Z=qC��q�Q��k�?W
=AI��C���                                    Bx���  �          @}p��p���r�\��p����C�\�p���o\)?8Q�A(z�C�3                                    Bx�
�  T          @��
�J=q��  �����C�c׿J=q��  ?\)@��\C�e                                    Bx�&  "          @�  ��  �q녿��p�C.��  �r�\?�@�C5�                                    Bx�'�  T          @q녿����e����(�C}n�����\��?�  AvffC|��                                    Bx�6r  �          @z=q��=q�h��>���@��Cy�H��=q�W�?�A�G�Cxp�                                    Bx�E  "          @tz��Q��Z=q=���?�p�Cs�H��Q��O\)?���A�ffCrp�                                    Bx�S�  �          @p�׿��H�U����33Cr�3���H�O\)?W
=ANffCr5�                                    Bx�bd  �          @tzῃ�
�fff�B�\�=p�C~\���
�`��?Y��AQ�C}��                                    Bx�q
  �          @p  ����k�����p�C��=����j=q?��A
=C��f                                    Bx��  T          @qG�?
=�`�׿5�4  C���?
=�c�
>���@���C���                                    Bx��V  �          @c�
?����4z�k��p��C�Z�?����=p���\)��{C��\                                    Bx���  �          @`��?����8�ÿ����(�C�S3?����H�þ���ڏ\C�`                                     Bx���  �          @i��?��R�1녿�=q��33C�{?��R�L(��s33�p��C��                                     Bx��H  T          @���>��
�n�R�����z�C�|)>��
���H�!G���C�E                                    Bx���  �          @���?&ff�e��(���z�C�q?&ff��=q��ff�e��C���                                   Bx�ה  �          @g��#�
�U�������C��3�#�
�Tz�?�AQ�C��\                                   Bx��:  �          @z�H�{�%�?��
A�Cb(��{�?�p�A�p�C\.                                    Bx���  �          @�  �8�ÿ�p�@G�B�CO(��8�ÿ@  @'
=B$�CB�=                                    Bx��  �          @{��'���\@�B��CU�R�'���  @0  B1�\CH�                                    Bx�,  
�          @xQ�� ����R?�33A��
C]��� �׿���@{BQ�CT\                                    Bx� �  �          @x����\�333?�33A��\Cf���\���@
=qBG�C`�H                                    Bx�/x  �          @vff�Q��Q�?�\)A���Ca)�Q��(�@   B!�\CW��                                    Bx�>  T          @xQ����� ��?�33A��Cd
��׿�@#�
B%z�C[(�                                    Bx�L�  �          @�  �)���)��@   A��Ca�)����Q�@,��B��CX+�                                    Bx�[j  �          @�ff����>{?��A��Ch�R�����@+�Bz�Ca.                                    Bx�j  �          @�G����H�Y��?�p�A�  Cp����H�.�R@*=qB��Cjc�                                    Bx�x�  T          @��׿�Q��Y��?ٙ�A���CpJ=��Q��/\)@(��B�Cj��                                    Bx��\  �          @���ff�Fff?��RA�Ck�f�ff��@5�B'=qCdaH                                    Bx��  �          @�
=��\�G
=?�
=A��Ci����\�p�@!�B�HCc                                    Bx���  �          @�p��	���O\)?�  A���Clk��	���(��@��B  Cf��                                    Bx��N  T          @�z���
�W�?��RA�
=Cn����
�5@(�B {Cj�                                    Bx���  �          @�p���(��Z�H?��RA��
Cp���(��8��@{B�RCk�R                                    Bx�К  "          @~{���J=q?:�HA)�Cj8R���2�\?�33A���Cf��                                    Bx��@  �          @tz����Mp�>W
=@J�HCm����?\)?�A��Ckٚ                                    Bx���  �          @n�R��Q��HQ�>u@n{CnG���Q��:=q?�
=A�G�Cl^�                                    Bx���  T          @{����
�I��?�\)A���Cp����
�%@��B�Ckn                                    Bx�2  �          @~{�����S33?�z�Ař�CvQ쿹���(��@%B"�RCq:�                                    Bx��  �          @~{��ff�S33?�  A��Cxzῦff�'
=@*�HB*33CsxR                                    Bx�(~  �          @z�H��{�J=q?�\)A��Cv�΅{��@/\)B233Cp�
                                    Bx�7$  �          @z=q��G��U�?��A�z�Cu�׿�G��/\)@BffCq33                                    Bx�E�  �          @z�H��
=�Y��?��
A�\)Cw8R��
=�5@G�B��CsO\                                    Bx�Tp  �          @x�ÿ���`  ?��
Au��Cy
����AG�@�
A��Cv\                                    Bx�c  �          @y�������`  ?h��AXz�Cw������C33?���A�(�Ct�{                                    Bx�q�  �          @y����(��XQ�?��\A���Cvn��(��4z�@  B
=Crh�                                    Bx��b  "          @x�ÿ�=q�`��?��HA�ffC|�
��=q�=p�@\)B�
Cy�H                                    Bx��  T          @z�H����a�?�{A��Cy𤿥��@  @	��Bz�Cv��                                    Bx���  T          @{���  �dz�?��Av{Cz����  �Dz�@ffB�\CwǮ                                    Bx��T  �          @x�ÿ�(��`  ?W
=AHz�Cw:ῼ(��C�
?��A�G�Ct\)                                    Bx���  �          @x�ÿ�{�`  ?z�HAi�Cx�쿮{�AG�@�A��
Cu��                                    Bx�ɠ  �          @o\)��ff�Vff?�  A�ffC|�\��ff�2�\@\)BffCyT{                                    Bx��F  �          @h�ÿ\)�J=q?�=qA�33C���\)� ��@\)B0�C��q                                    Bx���  �          @fff�8Q��P��?��RA�ffC�o\�8Q��'�@(�B+�C�\                                    Bx���  �          @o\)�:�H�b�\?=p�A8��C�0��:�H�G�?���A�(�C�q�                                    Bx�	8  �          @n�R��
=�c�
?n{Ah��C�� ��
=�E�@ ��BQ�C�)                                    Bx�	�  T          @o\)���aG�?��A�ffC��3���?\)@	��B=qC��                                    Bx�	!�  T          @g��0���Tz�?��RA�=qC�!H�0���0��@�RB\)C�                                    Bx�	0*  �          @j=q��33�W
=?aG�A^=qC{���33�9��?�33A���Cxff                                    Bx�	>�  �          @hQ쿬���U>��@�  Cx  �����B�\?��HA�33Cv�                                    Bx�	Mv  w          @g��p���Q�?+�A1p�C~\�p���9��?�
=A�C|\                                    Bx�	\  
�          @dzᾙ���\��?=p�A@��C��������A�?�ffA��C�4{                                    Bx�	j�  
�          @b�\�^�R�Tz�?��A��CG��^�R�=p�?�=qAظRC}�H                                    Bx�	yh  
�          @Fff�:�H�>{>�(�@��C��:�H�+�?�{A�33C~��                                    Bx�	�  T          @8Q�\)�1G�?�A)C�G��\)���?��A��C��\                                    Bx�	��  �          @AG��8Q��=p�?�\AG�C�=q�8Q��(��?�Q�A�
=C�f                                    Bx�	�Z  
�          @Q녾�
=�L��?\)A�\C�B���
=�5?���A��HC���                                    Bx�	�   T          @p  �5�h��?
=qAp�C�~��5�P��?�Q�A���C���                                    Bx�	¦  �          @dz�@  �\��>��H@��C��׿@  �Fff?�=qA��C�:�                                    Bx�	�L  �          @hQ쿵�S33�����Cv�ÿ��HQ�?�ffA�p�Cu�
                                    Bx�	��  �          @s33��\�L(��Y���R�\Cq\��\�S33>k�@fffCq�{                                    Bx�	�  
�          @tz῞�R�]p���ff�~=qCz33���R�g
=>\)@�Cz��                                    Bx�	�>  �          @u��Ǯ�Q녿��H���Ct���Ǯ�`  ��\)����Cv�                                    Bx�
�  "          @s33��\�;��5�-G�Ch���\�@  >�z�@�z�Ch��                                    Bx�
�  �          @s�
���HQ�(��z�Clu����J=q>�ff@���Cl��                                    Bx�
)0  �          @u���(��Mp��+��"=qCn����(��P��>�
=@ʏ\Cnٚ                                    Bx�
7�  �          @tz��\)�C�
��
=�ȣ�Ci�\)�AG�?�RA��Ciu�                                    Bx�
F|  "          @s�
�
=q�<(�������ffCi���
=q�HQ����CkW
                                    Bx�
U"  
�          @o\)��=q�@  �����Cn�Ϳ�=q�>�R?\)Az�Cnh�                                    Bx�
c�  �          @Dz�E���
��\)��(�C{s3�E����?333A��Cz�)                                    Bx�
rn  �          @W
=>����H@O\)B���C��3>�?�R@N{B�#�BU��                                    Bx�
�  �          @j=q?���B�\@Tz�B�p�C���?��?n{@L��B}{B                                      Bx�
��  
�          @x��?�Q�\@j�HB�k�C��?�Q�?\(�@eB�u�B
=                                    Bx�
�`  �          @]p�?}p��fff@H��B���C��f?}p�>L��@P��B���A6ff                                    Bx�
�  �          @3�
�����"�\�����Cw#׿����#33>��AQ�Cw5�                                    Bx�
��  T          @2�\�����{�#�
�V{Cu녿����"�\>aG�@��Cv��                                    Bx�
�R  �          @k���{�Q녾�� ��Cs����{�HQ�?z�HAyCrٚ                                    Bx�
��  �          @X�ÿ�G��5�>�33@�=qCn{��G��"�\?��A���CkG�                                    Bx�
�  T          @aG���{�9��?��A��Cm@ ��{�"�\?��RA�
=Ci�                                     Bx�
�D  �          @w
=��
=�Vff?fffAW\)CsQ��
=�6ff?�p�A�
=Cok�                                    Bx��  �          @p�׿�(��U?fffA^=qCvJ=��(��5�?�p�A���Cr�f                                    Bx��  �          @a녿�33�Dz�?h��Ar�HCuzῳ33�$z�?�33B
=CqxR                                    Bx�"6  �          @>�R���
�$z�?��A��Cx0����
��\?�33BCsO\                                    Bx�0�  �          @&ff��  �\)?\(�A���Cu�f��  ��?�=qB�Cq�                                    Bx�?�  �          @`  ��33�J=q?8Q�A@��Cy�ÿ�33�.�R?�G�A���Cw\                                    Bx�N(  �          @l(����H�J=q?=p�A9Cq�\���H�-p�?�\A�Cm�=                                    Bx�\�  �          @l�Ϳ˅�K�?E�AB�HCsn�˅�.{?�A��Co�R                                    Bx�kt  �          @e�!G��U?���A��\C���!G��0  @p�B{C���                                    Bx�z  �          @i���G��X��?��A��HC����G��3�
@(�B�C~�                                     Bx���  �          @aG�����L(�?���A�p�C�&f����!�@�B*�C���                                    Bx��f  T          @l�Ϳ����U�?z�HAxQ�Cz0������1�@z�B	�Cv��                                    Bx��  "          @x�ÿ�{�c33?O\)A@��Cy\��{�C33?��RA�z�Cu��                                    Bx���  
�          @|�Ϳ���h��?=p�A-Cz(�����J=q?��HA�RCwff                                    Bx��X  �          @��׿��c�
?E�A1Ct�����Dz�?��HA��Cq}q                                    Bx���  �          @{���  �j=q?s33A`��C~�R��  �Fff@
�HBp�C|�                                    Bx��  T          @|(�����c33?�G�Ao33CyY�����>�R@(�B�Cu                                    Bx��J  T          @|(����P  ?�@�ffCl�����7
=?У�A�z�Cic�                                    Bx���  �          @|(���\�HQ�?
=q@�\)Ci�
��\�/\)?�{A�Cf&f                                    Bx��  �          @{��.{�0��>��@��HCaff�.{��H?�z�A��RC]�f                                    Bx�<  "          @y������<(�?^�RAO�
Cf��������?���A㙚Ca��                                    Bx�)�  
�          @~�R�A��=q>�  @p��CZp��A��
=q?��A�{CW}q                                    Bx�8�  T          @\)�@  �"�\?\)AG�C\T{�@  ��?�A�CX�                                    Bx�G.  T          @����I���=q?8Q�A$(�CYxR�I���   ?��
A��CTu�                                    Bx�U�  T          @����H���$z�?.{A�C[@ �H���
=q?��A�Q�CV��                                    Bx�dz  
�          @�G���Z�H��
=��  Cn�{��Vff?Q�A<(�Cn�                                    Bx�s   T          @����Q��Z=q�\)���RCm�R�Q��N�R?�=qAyCl�)                                    Bx���  �          @�(�����Tz�W
=�;�Ci������K�?}p�A\��ChW
                                    Bx��l  T          @���(��aG���{��(�Cn0��(��Z=q?p��AQ�CmY�                                    Bx��  �          @��R��\�j=q��{��ffCp����\�b�\?}p�AY��Cp�                                    Bx���  �          @�\)�  �e���  �Z�HCm�{�  �[�?�ffAd��Cl�q                                    Bx��^  T          @�\)�����q녾8Q��p�Cs�����e?�Q�A��Cr�                                    Bx��  �          @�\)��{�xQ�<#�
>8Q�Cw����{�g�?�z�A��RCv
=                                    Bx�٪  T          @����{�w
=>�
=@�ffCwff��{�\��?��
Aȏ\Cu                                    Bx��P  �          @�{�%��QG����
���HCg���%��J�H?c�
AD(�Cf޸                                    Bx���  T          @�
=����o\)>�{@��CsO\����W
=?�A���Cp��                                    Bx��  
�          @��Ϳ�  �u>�ff@�{Cx���  �Z�H?�A��HCvQ�                                    Bx�B  "          @�(������y��?
=A33C|޸�����Z�H?��HA�Cz�H                                    Bx�"�  �          @\)��H�J�H�\)�G�Ch����H�@  ?��\Amp�Cg!H                                    Bx�1�  T          @z�H�ff�Q�<#�
=���CmY��ff�C33?���A��Cks3                                    Bx�@4  �          @p�׾�
=�R�\?z�A�C�Q��
=�6ff?޸RA�p�C�                                    Bx�N�  �          @X��@0�׿h��?޸RA�{C��@0�׾W
=?���B�C�ٚ                                    Bx�]�  T          @]p�@3�
�E�?��BG�C�^�@3�
��@�BQ�C��q                                    Bx�l&  �          @S�
@(�ÿ333?���B  C��3@(��<��
?���B
=>��                                    Bx�z�  
�          @Tz�?�=q�
�H?�z�A��C��?�=q��z�@B5Q�C�1�                                    Bx��r  T          @Z=q?�  �.{?˅A��
C�U�?�  ����@�RB<�C�^�                                    Bx��  T          @p  ?�33�:=q?��A�C��)?�33� ��@4z�BD�C�q�                                    Bx���  �          @r�\?�  �8Q�?�Q�A��C���?�  ���H@7�BEQ�C���                                    Bx��d  w          @o\)?��%�@�B
33C�|)?���{@:�HBN  C��                                    Bx��
  �          @g
=?��R��
?�
=B
=C�b�?��R��
=@)��B<��C�'�                                    Bx�Ұ  �          @g
=?�z��&ff?˅A��C�#�?�z��=q@��B+�C�'�                                    Bx��V  �          @b�\?�Q��*�H?�=qAԏ\C�'�?�Q���@p�B0��C���                                    Bx���  �          @\(�?޸R��@�\B�C���?޸R���H@+�BNz�C��R                                    Bx���  T          @h��>�ff�.�R?ٙ�A�z�C���>�ff��33@%BTQ�C���                                    Bx�H  �          @]p����H�)��>Ǯ@��HCixR���H��
?�{A��Ce��                                    Bx��  �          @{��B�\��
������ffCY&f�B�\��\?   @�p�CX�                                    Bx�*�  T          @�33�i����  ?�\@��CM�H�i����
=?���A�z�CIk�                                    Bx�9:  �          @����1G��6ff?�@��Ca�=�1G��(�?ǮA�33C]^�                                    Bx�G�  T          @�Q��z��Mp�?\)@��Cj&f�z��1G�?�p�A�Q�Cf                                      Bx�V�  �          @����>�R�G�?��A��CYG��>�R�˅@ffA��HCP!H                                    Bx�e,  "          @z=q�#�
����@�HB  CT��#�
��@6ffB=��C@��                                    Bx�s�  �          @}p��S�
���R?�\AԸRCH���S�
��@ffB ��C<0�                                    Bx��x  T          @|(��)���.{?}p�Ai�Ca��)�����?���A�(�CZ��                                    Bx��  T          @}p��5��*=q?(��A\)C_L��5��p�?У�A��CZ
=                                    Bx���  �          @|(��W���33��33��(�CQp��W����>���@�ffCQL�                                    Bx��j  �          @tz��3�
�\)����HC?8R�3�
���׿�����CN33                                    Bx��  �          @��
�{�.�R��z���ffCf���{�(Q�?G�AK
=Ce�f                                    Bx�˶  �          @�p��K��B�\��G����C_�=�K��6ff?���A`Q�C]�                                    Bx��\  �          @�{�U��<(�=�G�?�Q�C]n�U��*�H?�p�A�z�CZ�q                                    Bx��  
�          @���Z=q�,�ͼ���Q�CZaH�Z=q� ��?�G�AW
=CXJ=                                    Bx���  T          @��H�K��;�>���@�  C^���K��%?�33A�33C[
                                    Bx�N  �          @�33�Tz��.�R?\)@�p�C[n�Tz���
?���A��CV                                    Bx��  T          @�Q��S33�'�?
=@�ffCZk��S33�(�?�ffA�33CU��                                    Bx�#�  �          @���S33�%?�@�{CZ0��S33�
�H?\A���CUff                                    Bx�2@  T          @�G��Tz��)��?��AG�CZ���Tz��p�?�=qA���CU�H                                    Bx�@�  
�          @����U�*=q>�ff@�G�CZ���U��?���A��
CV^�                                    Bx�O�  �          @�  �P  �(��?333AQ�C[��P  �
=q?�A��CU��                                    Bx�^2  �          @�  �R�\�(Q�?�@��HCZ�)�R�\�{?�G�A�{CU��                                    Bx�l�  �          @�ff�N{�)��>�(�@��HC[s3�N{�G�?�A���CW=q                                    Bx�{~  "          @����.{�<��>aG�@L(�Cc\)�.{�(Q�?�{A��C`�                                    Bx��$  �          @�=q�3�
�:�H>��R@��Cb#��3�
�$z�?�
=A��RC^p�                                    Bx���  T          @��H�?\)�1G�>k�@S33C^�{�?\)�p�?�ffA��\C[n                                    Bx��p  �          @��
�7��9��>�Q�@��CaT{�7��!�?�(�A��\C]^�                                    Bx��  �          @��
�*�H�9��?�ffAn{CcaH�*�H���@ffA�=qC\5�                                    Bx�ļ  �          @�=q����P��?B�\A-CkL�����,��?�p�A��Cf
                                    Bx��b  �          @�z��B�\�1G�>�=q@p  C^Q��B�\�(�?��A���CZ��                                    Bx��  �          @�����\�O\)?\)A ��Cj�R��\�0��?��Aԣ�Cf@                                     Bx��  �          @�  �޸R�Y��?z�HAf�HCr�׿޸R�.�R@\)B
�
Cm��                                    Bx��T  
�          @��Ϳ�=q�`��?�A��Crz��=q�0��@p�Bz�Cl}q                                    Bx��  �          @��H��R�Vff?(�A	��ClT{��R�5�?��A���CgǮ                                    Bx��  �          @��(��U>���@�Q�Ci�)�(��<(�?У�A�z�CfG�                                    Bx�+F  �          @��0���[�>�
=@��Cg=q�0���>�R?�  A���Cc@                                     Bx�9�  �          @�z��>{�H��?=p�A
=Cb�\�>{�%�?�
=A�z�C]                                    Bx�H�  �          @�=q�G��:=q?8Q�A��C^��G��Q�?���AƏ\CYG�                                    Bx�W8  �          @���C�
�E>�\)@i��CaO\�C�
�.{?��RA��
C]��                                    Bx�e�  �          @�ff�L(��E�>���@y��C_��L(��,��?�G�A���C\0�                                    Bx�t�  �          @����QG��G�>u@C�
C_���QG��0��?�(�A��C\+�                                    Bx��*  T          @�  �S33�C33�\)��ffC^��S33�6ff?���Aa��C\�
                                    Bx���  "          @�z��HQ��?\)?z�@��HC_�3�HQ�� ��?�p�A��RCZ��                                    Bx��v  �          @���N{�7�?J=qA&=qC]���N{��
?��A���CW��                                    Bx��  �          @���P  �=p�>�G�@�Q�C^W
�P  �!�?���A�CY�                                    Bx���  
�          @���P���3�
?333AQ�C\�P����?�\A�ffCW�                                    Bx��h  �          @�\)�Tz��Q�?n{AL��CW��Tz��?�A�
=CP��                                    Bx��  �          @�
=�p�׿��?�@��HCK���p�׿��
?��A���CF��                                    Bx��  �          @��y����p�>\)?�CH���y����ff?:�HA ��CFh�                                    Bx��Z  T          @�=q�mp�� �׾��R��\)CPp��mp����H?�@�RCO�H                                    Bx�   �          @����l(��
=�\��CT���l(���
?(�A (�CT                                      Bx��  �          @���n�R�33�����z�CS���n�R�G�?\)@���CSB�                                    Bx�$L  �          @�(��g
=�(���G����CV
�g
=�=q?
=@�Q�CU�                                     Bx�2�  
�          @���e��!G������
CW��e��!�?�\@���CW8R                                    Bx�A�  �          @��e�   �333�CV�
�e�$z�>\@�ffCW��                                    Bx�P>  �          @�\)�l���(��.{��CUs3�l��� ��>\@��CV!H                                    Bx�^�  �          @�p��k����#�
���CT�\�k���H>Ǯ@�=qCUc�                                    Bx�m�  
�          @�33�fff�녿aG��:ffCTJ=�fff�(�>�?�\CV�                                    Bx�|0  �          @�(��n�R�p��
=��{CR���n�R�  >�p�@��CS�                                    Bx���  �          @�p��xQ���\���H���
CO� �xQ���
>���@�(�CO��                                    Bx��|  �          @�z��s33��ÿ   ��33CQff�s33�
=q>�
=@�G�CQ��                                    Bx��"  �          @�Q��i���
�H�������HCR�R�i�����?�@�ffCRaH                                    Bx���  �          @�{�j�H���5���CN�{�j�H���H=�G�?���CP(�                                    Bx��n  T          @��g��\����G�CJ���g���(��(���ffCP�)                                    Bx��  "          @��k���33������(�CH�{�k����Ϳ+��(�CN�H                                    Bx��  "          @~{�h�ÿ��Ϳ�{��(�CD�\�h�ÿ��H���� Q�CI޸                                    Bx��`  �          @p  �8Q��;�?s33A��C�B��8Q��G�@B)�HC��H                                    Bx�   �          @qG�?�
=�'
=?�=qA�C�1�?�
=��{@0��B>��C��                                    Bx��  �          @���?�ff�=p�@   A�=qC���?�ff����@C�
BG{C�3                                    Bx�R  �          @���?�G��c�
?�(�A�p�C��q?�G��!G�@C33B=
=C�H�                                    Bx�+�  �          @�(�?����k�?�  A�ffC�Q�?����.{@9��B2��C�                                    Bx�:�  �          @��H?#�
�w�?��A�p�C���?#�
�B�\@)��B"(�C��
                                    Bx�ID  �          @{�?��j�H?��HA�Q�C��?��4z�@(��B*(�C�7
                                    Bx�W�  �          @��?���dz�?�Q�A�p�C���?����H@P��BR�C��                                    Bx�f�  �          @��H?8Q��s33?�A�ffC�` ?8Q��=p�@*=qB$�\C���                                    Bx�u6  �          @��>�G��s33?��A�33C�L�>�G��:=q@1�B-p�C�N                                    Bx���  �          @��H=�G��w
=?�=qA��C��
=�G��<(�@5�B/�RC��                                    Bx���  T          @��\<��
�p��?��
A���C�0�<��
�1G�@>{B<G�C�AH                                    Bx��(  T          @��׽�G��u�?��HA��
C�'���G��=p�@-p�B)��C���                                    Bx���  �          @~{�����u�?��
Aqp�C�H������A�@#�
B �C�R                                    Bx��t  �          @\)�=p��mp�?�p�A�  C�XR�=p��5@,(�B)�CY�                                    Bx��  �          @z�H�J=q�`  ?�(�A�C��ͿJ=q�)��@%B,Q�C}�                                     Bx���  
�          @j�H?!G����@!�B6�C�N?!G�����@S�
B�(�C��{                                    Bx��f  "          @h��>�\)�E?���A��HC��>�\)�
=@0��BQ��C���                                    Bx��  �          @l�;8Q��hQ�>�Q�@��
C��q�8Q��HQ�?�\)A��\C�e                                    Bx��  �          @�  �B�\�z=q?G�A5�C��R�B�\�L��@Q�B=qC�G�                                    Bx�X  �          @u��z��S�
�O\)�FffCs^���z��U?.{A$��Cs�{                                    Bx�$�  �          @z=q���J=q��ff�~�\Cn����S�
>���@��RCo��                                    Bx�3�  �          @|(���33�[��p���\��CtaH��33�`  ?!G�A
=Ct�{                                    Bx�BJ  �          @�G�����qG��&ff���C{{����k�?��
Al  Cz�                                    Bx�P�  �          @�������:=q������Cf������L��<�>�ffCiW
                                    Bx�_�  "          @�  ����:�H��Q�����Ce�����I��>8Q�@#�
Ch\                                    Bx�n<  "          @��0���6ff�����{Ca���0���HQ�=u?c�
Cd�H                                    Bx�|�  T          @������J�H������\)Ci�����]p�=�?�z�Ck\)                                    Bx���  �          @�G���
=�h�ÿ�Q�����Cr
=��
=�s33?   @�Cs�                                    Bx��.  T          @�=q�Q��Z=q�����  Cn��Q��o\)=��
?�{Cph�                                    Bx���  T          @�ff���L(�����G�Cl�3���fff����Q�Cp                                    Bx��z  "          @�33����HQ�����Cn�׿���g�������  Crn                                    Bx��   "          @�
=�����Mp������{Cn�Ϳ����n{���R��Q�Cr\)                                    Bx���  �          @�Q��G��S�
���
��z�Cuk���G��h��=u?Y��Cwu�                                    Bx��l  "          @|(��޸R�H�ÿ�
=��33Cq��޸R�\��=��
?�p�Cs=q                                    Bx��  �          @��ÿ�Q��L(����R���Cn����Q��aG�=L��?5Cq�                                    Bx� �  T          @��\�{�J�H������G�Cj���{�[�>B�\@)��Cm{                                    Bx�^  �          @�  �\)�G
=��G����
CgY��\)�]p�    <�CjB�                                    Bx�  �          @�z��
�H�P�׿����ffClL��
�H�`��>u@W
=Cn33                                    Bx�,�  "          @�����Q��G���(��ȣ�Cn\��Q��c33�8Q��#�
CqW
                                    Bx�;P  T          @�33�ff�@�׿��H����Ch��ff�Vff    =uCj�                                    Bx�I�  �          @w
=�Ǯ�H�ÿ����\Cs�)�Ǯ�\��=���?�G�Cu��                                    Bx�X�  �          @s33��\�=p���p���33Co{��\�S�
���ǮCq�\                                    Bx�gB  "          @i����
=�C�
������Cx녿�
=�Tz�>B�\@Dz�Czh�                                    Bx�u�  �          @mp��n{�S33��Q�����C~:�n{�_\)>Ǯ@��HC
=                                    Bx���  "          @w
=>k��k�?�ffA}�C�>k��5@#�
B'�
C�Ff                                    Bx��4  T          @l�;u�c�
?uAp��C�{�u�1G�@�B$��C��=                                    Bx���  �          @s�
�!G��hQ�?n{Aa��C��!G��6ff@�B\)C��)                                    Bx���  �          @y���333�qG�>�G�@�  C����333�L(�@�
B  C�Ф                                    Bx��&  �          @{���Q��n�R=�?���C|B���Q��R�\?�G�A��Cz\                                    Bx���  �          @}p���G��l(�����{C{33��G��b�\?�z�A��CzxR                                    Bx��r  �          @�=q�z��\)��33���C�ٚ�z��n�R?��HA��C���                                    Bx��  "          @~�R�.{�w��#�
��C���.{�p  ?�33A�{C��                                     Bx���  
�          @mp��k��U>���@��C~�)�k��7
=?�G�A��C|(�                                    Bx�d  �          @j�H�����g�=�\)?�ffC�^������N{?�A���C�
=                                    Bx�
  T          @s�
��\)�mp�?.{A%C�Ф��\)�AG�@��B��C�P�                                    Bx�%�  �          @s�
�����U?��RA�p�C{�쿌����@4z�B>��Ct                                    Bx�4V  �          @z=q��Q��`  ?s33Aa��Cw�
��Q��-p�@�HB��Cq�                                    Bx�B�  T          @\)�s33�tz�>k�@S�
C�  �s33�S�
?�z�A�  C}�R                                    Bx�Q�  �          @�  �J=q�w������33C�>��J=q�k�?��A���C���                                    Bx�`H  �          @j�H���
�O\)��Q����RC|T{���
�[�>\@�(�C}B�                                    Bx�n�  �          @y���\�c�
    �uCv�H�\�K�?˅A�z�Ctz�                                    Bx�}�  �          @x�ÿ�ff�h��?5A*�\C}�3��ff�;�@G�Bz�CzW
                                    Bx��:  "          @mp��G��`�׾k��l��C��H�G��O\)?�\)A��HC�@                                     Bx���  
�          @c33�����Z�H�&ff�+�C�<)�����Vff?uA~=qC�,�                                    Bx���  �          @W���{�P�׿:�H�JffC����{�O\)?O\)A_�
C��                                    Bx��,  �          @R�\�Y���HQ����G�C~��Y���3�
?��A��HC}�                                    Bx���  
�          @Dzᾅ��;��Tz��|��C�t{����@  ?z�A/33C��H                                    Bx��x  	.          @O\)�\)�H�ÿ5�Lz�C����\)�G�?G�Aa�C��)                                    Bx��  
�          @J�H�\)�B�\�aG�����C��׾\)�G
=?
=A,Q�C���                                    Bx���  T          @>�R=#�
�
=��\���C�q�=#�
�:=q��\��C�\)                                    Bx�j  
�          @@�׾8Q��,(���33���RC�!H�8Q��:�H>L��@tz�C�G�                                    Bx�  �          @P  �����N{�����C�T{�����?\)?�(�A��C�                                      Bx��  �          @qG��c�
�Vff?z�HA|��C{�c�
�"�\@��B&�RCz�H                                    Bx�-\  �          @�G����R�e�?�z�A�=qCz�����R�z�@U�BN�
Cq��                                    Bx�<  "          @�����
=�fff?�A�
=CxE��
=��@Q�BG(�Cn��                                    Bx�J�  �          @��\�\�hQ�?޸RA�z�Cw@ �\�(�@L��B@33Cn
=                                    Bx�YN  �          @�33�Ǯ�s�
?���A�  Cw�=�Ǯ�0��@=p�B,=qCp��                                    Bx�g�  "          @�=q��z��p  ?��\A�p�Cv!H��z��0��@5B%�RCn�3                                    Bx�v�  
�          @��
���g
=?�
=A�\)Cs\)���(�@I��B8�Cih�                                    Bx��@  
Z          @��\��
=�`  ?�33A�{Cq#׿�
=�ff@Dz�B5�Cf�                                    Bx���  �          @�33�����[�?�A�
=Cpn�����{@K�B<ffCdǮ                                    Bx���  �          @������J�H?��
A��RCh޸����   @A�B0�C[�
                                    Bx��2  �          @�=q����[�?�\)A�z�CmQ������H@2�\B!�
Cc��                                    Bx���  T          @�=q���^{?��A�ffCo������R@0��B"��Cf�3                                    Bx��~  �          @��ÿ�=q�dz�?�Q�A��Cv\��=q���@H��B>z�ClxR                                    Bx��$  �          @��׿�(��n{?��HA���Cxk���(��(Q�@@��B3�HCpǮ                                    Bx���  T          @����  �z�H?�G�A���C�q��  �9��@;�B.�C{                                    Bx��p  �          @�  ����|(�?���Ax��C~������>{@4z�B'33CzW
                                    Bx�	  �          @�z��G��u?(�@�ffCr8R�G��HQ�@�
A�ffCm�                                    Bx��  �          @�����j=q?�=qAb=qCn������0  @)��B��CgT{                                    Bx�&b  �          @�z����mp�?�{A�G�Cs(�����*=q@:�HB'CjǮ                                    Bx�5  T          @��׿�(��q�?��HA���Cx�R��(��2�\@4z�B'(�Cr33                                    Bx�C�  T          @�������~�R?@  A#\)C~!H�����K�@   B33Czn                                    Bx�RT  T          @��׿(���q�?��
A��C���(���!G�@U�BO��CL�                                    Bx�`�  
�          @��ÿ���mp�@G�A�=qC�e����ff@aG�B]�\C�H                                    Bx�o�  �          @�=q�k�����?��Av=qC��)�k��C�
@8��B(C}E                                    Bx�~F  �          @�  �.{��p�>�=q@l(�C�Z�.{�dz�@�A�C���                                    Bx���  
�          @�=q�����e�?�z�A�Q�C{c׿���� ��@:�HB9\)CtY�                                    Bx���  �          @�z῕�l(�?�(�A��C|ff���p�@O\)BG�RCt�{                                    Bx��8  2          @�{�u��G��s33�R�RC�P��u��Q�?��Ah��C�N                                    Bx���  �          @�  ��ff�W
=��\����C�+���ff�{��u�`  C��
                                    Bx�Ǆ  T          @�  ���'
=��R�Cm\)���^{�z�H�c�
CtB�                                    Bx��*  �          @�������U���z���(�Cv������n{=�Q�?��Cx�q                                    Bx���  �          @��>��N�R�����
C�.>��z=q���H��33C���                                    Bx��v  T          @��\�aG���
=�Y���6�RC�y��aG���(�?�(�A��C�p�                                    Bx�  T          @��R>L�����Ϳ!G���C�S3>L����{?�G�A�=qC�e                                    Bx��  d          @�
=>����(��L���%G�C��
>����  ?���A���C���                                    Bx�h  
�          @�Q�?#�
���
�Tz��)��C�'�?#�
��  ?���A��RC�Ff                                    Bx�.  �          @���>�\)��z�޸R��Q�C���>�\)��
=>���@��C��                                    Bx�<�  �          @�ff��p��U�,(���HC�� ��p����R�O\)�-�C��                                    Bx�KZ  
�          @��
>k��\)�����\)C��=>k��p  ?��HA�=qC��f                                    Bx�Z   �          @���>�(����\�����C���>�(��y��?�33A�33C�,�                                    Bx�h�  T          @�ff>�  ���
�#�
��C��f>�  �w
=@33A�p�C��q                                    Bx�wL  �          @�{��Q���{������(�C��;�Q����\?J=qA$(�C��H                                    Bx���  �          @�  ���R�y�����
��{C|J=���R����>�=q@^{C}�                                    Bx���  �          @�  �����j�H�	������Cxp�������  �.{�
=C{+�                                    Bx��>  "          @�ff����S�
�   �	�Cu  �����33�#�
�G�Cy^�                                    Bx���  "          @�z�L���S�
�0����C�@ �L����
=�\(��5��C��=                                    Bx���  �          @��\�.{�L���5��#z�C���.{����z�H�Qp�C�W
                                    Bx��0  
�          @�����{��u�B�\CuT{���_\)?�A�  Cr��                                    Bx���  "          @�ff���\��zῑ��p(�C�쿂�\��{?uAH  C�%                                    Bx��|  T          @�
=��Q��r�\����ʣ�C|���Q���\)>\)?�{C~=q                                    Bx��"  �          @�  �h���~{��ff����C��\�h����33>���@xQ�C��                                    Bx�	�  
�          @��׿�����33��  ���C{�ÿ�����ff?Y��A-�C|�                                    Bx�n  
�          @��
��p��4z�@P��B6�
Cr=q��p��n{@��RB��fCT�                                    Bx�'  "          @���  �\(�@�A�(�Cl�\�  �G�@`  BDC]�H                                    Bx�5�  
�          @�����
�\)?�  A�G�Cv����
�333@N{B0��Cm�)                                    Bx�D`  
�          @��H�	���r�\�(���p�CpaH�	���g�?��\A�ffCo=q                                    Bx�S  
�          @���
=�{�����fffCq�=�
=��  ?^�RA-Cr5�                                    Bx�a�  
�          @�33������������  Cy�q����xQ�?�p�A��CxT{                                    Bx�pR  T          @�z��
=��Q�?
=@�\Cxs3��
=�^{@#33B33Ct{                                    Bx�~�  "          @�zῸQ���p�?�A�z�Cz�H��Q��>�R@O\)B133Ct(�                                    Bx���  T          @��Ϳ����\)?�Q�A�  Cx������,��@X��B=�\Co�)                                    Bx��D  T          @�������  ?&ffA�Cz
����Z�H@'
=BG�Cu                                    Bx���  "          @�{��  ��{?aG�A.�RC�����  �^{@8��B�RC}��                                    Bx���  �          @����!G����
?��AR�\C�  �!G��a�@J�HB%�HC��{                                    Bx��6  �          @�33�@  ���\?��RAo
=C�` �@  �Z�H@Q�B,z�C��)                                    Bx���  �          @�G�>�Q���33?�{AX��C�<)>�Q��`  @K�B(z�C��                                    Bx��  �          @�
=�`���7
=@�A�p�C[8R�`�׿��@J=qB
=CK��                                    Bx��(  "          @����(��b�\@�A��
Cwc׿�(�� ��@l��B`�Ci��                                    Bx��  T          @�ff����{�@!G�B�HC�������R@��HBsz�C��                                    Bx�t  �          @�  ?�  �S33@FffB%
=C�W
?�  ��z�@���B���C���                                    Bx�   �          @���?����:=q@Z=qB7p�C�7
?��ÿh��@�z�B���C���                                    Bx�.�  �          @��?�Q��0  @Z�HB5�HC��R?�Q�B�\@��\B��C�E                                    Bx�=f  �          @�G�?��
�\)@j�HBH�\C��?��
���@��B���C�xR                                    Bx�L  T          @���?�p��C�
@AG�B"�
C��H?�p���(�@��
B{G�C�g�                                    Bx�Z�  T          @��H?���j�H@�RA�RC�ٚ?���Q�@n{B`\)C��                                    Bx�iX  2          @��H?5�\��@ffA�(�C���?5� ��@`  BjC��q                                    Bx�w�  v          @��?�ff���@ ��B{C���?�ff�u@5�BdC�0�                                    Bx���  "          @��H@Q��G�@p  B\\)C�Q�@Q�>��H@�Q�Bu��AM                                    Bx��J  2          @��@{��ff@UB>33C��3@{=#�
@r�\Bc��?p��                                    Bx���  v          @�=q@���G�@U�B:�C���@���#�
@x��Bi=qC�&f                                    Bx���  �          @�@�
�333@G
=B'(�C�,�@�
�p��@�=qBs��C��\                                    Bx��<  �          @�\)?����Z=q@.{B�RC�3?��ÿ�z�@���BjG�C��
                                    Bx���  T          @��@   �c�
@{A�z�C��\@   ��\@j�HBPz�C�Ff                                    Bx�ވ  "          @�Q�?�(��HQ�@�A�33C��?�(��У�@`��BW�RC�4{                                    Bx��.  �          @�  ?!G��J�H?���AθRC���?!G�� ��@:=qBX33C��                                     Bx���  T          @��H����*=q@!G�B\)Cfn��Ϳ�{@`  B[Q�CN�f                                    Bx�
z  "          @��ý�\)���\?�33A�{C�|)��\)�(��@i��BXQ�C�4{                                    Bx�   
(          @�Q�
=�S�
@8Q�B"(�C���
=��  @�(�B�p�Cxp�                                    Bx�'�  
�          @���>W
=�k�@#33B
��C���>W
=��(�@���B~�RC�{                                    Bx�6l  
�          @������?��A�{C�33���?\)@J�HB:�C���                                    Bx�E  �          @����\�s�
?��A�  C�/\��\�=q@_\)B[
=C��                                    Bx�S�  �          @�=q�O\)���?�
=A�z�C��q�O\)�@  @S�
B;(�C~��                                    Bx�b^  T          @�
=��ff��33?@  A�C���ff�[�@1�B�HC�G�                                    Bx�q  T          @�  �#�
��ff?&ffAz�C��q�#�
�dz�@.�RB�HC���                                    Bx��  "          @�녾.{��Q�?�R@�(�C���.{�hQ�@/\)BG�C���                                    Bx��P  �          @��R�#�
������=qC�� �#�
�{�@�
A�G�C��
                                    Bx���  "          @�ff������(�����Q�C������x��@�A��C��                                    Bx���  "          @�\)=L����ff�����u�C�S3=L������?�A˅C�\)                                    Bx��B  
�          @�  ������p�=�Q�?�{C�aH�����s�
@\)A�
=C�H�                                    Bx���  
�          @��\���H��(�>.{@�C~����H�o\)@�\A�C|�                                    Bx�׎  �          @�{�!G���  ����ffC�~��!G��c33?���A�
=C��\                                    Bx��4  
�          @�Q�.{�}p�?��A�z�C��׾.{�3�
@EB>ffC�B�                                    Bx���  �          @���?#�
��Q�?�Q�AvffC�G�?#�
�HQ�@G�B1G�C���                                    Bx��  �          @�녿����(�����UCxY�����n�R?�ffA��RCvB�                                    Bx�&  �          @��@(��I��@�
A�Q�C�^�@(���{@c�
BRz�C��f                                    Bx� �  �          @�33?��H�x��?��RAЏ\C��f?��H�=q@h��BV  C�W
                                    Bx�/r  �          @��
?�  ��ff?��HA�33C���?�  �<(�@U�B;��C�XR                                    Bx�>  T          @��\>�z���z�?���Ae�C���>�z��QG�@H��B.��C�}q                                    Bx�L�  T          @�=q�u�~�R?��A�C�=q�u�2�\@J�HBBG�C�}q                                    Bx�[d  �          @�(�������?���A��
C}� ���4z�@J�HB8p�Cwu�                                    Bx�j
  T          @�p��������?
=@���Cv������U�@#33B33Cq��                                    Bx�x�  �          @�\)�����{?�G�Aw\)Cv޸����A�@J=qB'�RCoh�                                    Bx��V  �          @�Q�������H>��H@���Cv޸�����b�\@#�
B��CrW
                                    Bx���  T          @��R�У���z����Cy�f�У��xQ�@�
A��HCw8R                                    Bx���  �          @��\���R��������=qC|𤿞�R���H?
=q@�=qC~�                                    Bx��H  �          @��8Q��Mp��:�H�&�\C���8Q���\)�u�J=qC�&f                                    Bx���  "          @�33�k����
?��A�(�C�ff�k��)��@j�HBW��C��                                    Bx�Д  
�          @�(�>�z��i��@333B�C�=q>�z���@��B��RC���                                    Bx��:  �          @�������?�{A��
C��\���;�@`��BG  C�                                      Bx���  �          @�p��u��Q�>�p�@�G�C�H�u�o\)@"�\B�C��                                    Bx���  �          @���J=q��z�?L��A#
=C�� �J=q�Z�H@7�B��C�w
                                    Bx�,  �          @�(��W
=��{?��HAs33C����W
=�QG�@P  B3
=C�                                      Bx��  �          @��
?
=q����?!G�@���C�` ?
=q�g�@2�\B{C�5�                                    Bx�(x  �          @�=�\)����?=p�A�
C�p�=�\)�dz�@7�Bp�C��                                    Bx�7  �          @�ff���
���?c�
A.=qC�� ���
�dz�@C33B"�C��
                                    Bx�E�  �          @�ff����Q�?��\A|(�C��=���R�\@UB5��C���                                    Bx�Tj  
�          @�{>������
?У�A���C�"�>����?\)@e�BG��C�q                                    Bx�c  �          @��?&ff���?���A���C�h�?&ff�)��@p  BX  C��R                                    Bx�q�  �          @�z�?�  ����?�=qA�\)C���?�  �&ff@eBL��C��q                                    Bx��\  �          @��>�������?�  A�G�C���>����6ff@h��BN�RC���                                    Bx��  �          @��?(��a�@1�B��C��?(���Q�@��B�33C��
                                    Bx���  T          @�(���=q�|��@�A�
=C����=q�  @�Q�Bq�C��f                                    Bx��N  �          @��
?�=q�G�@Mp�B1  C���?�=q��{@��HB���C�                                    Bx���  T          @��>Ǯ�}p�@A�G�C���>Ǯ���@q�Bd�RC���                                    Bx�ɚ  �          @��R�c�
���H>\@�p�C��H�c�
�s33@'
=B(�C�e                                    Bx��@  �          @�p��(����H?�\@ə�C�=q�(��n�R@.{B
=C�b�                                    Bx���  �          @��R�=p����?�@�  C�t{�=p��o\)@0  BQ�C�j=                                    Bx���  �          @���^�R��G�?�G�AD��C����^�R�\(�@HQ�B%�\C��                                    Bx�2  "          @�{��\�*=q@FffBCffC��\��\�E�@�Q�B�{Cl^�                                    Bx��  T          @��Ϳ#�
����@�z�B�CuT{�#�
?z�H@�  B��)B��
                                    Bx�!~  
�          @��;���:�H@&ffB%Q�C�e��׿��R@mp�B���Cy!H                                    Bx�0$  �          @�p��\(��2�\@g
=BLQ�C|ٚ�\(���R@���B��\CW޸                                    Bx�>�  T          @��ÿ���HQ�@5�BG�CwT{������@�Q�B�u�Ca
=                                    Bx�Mp  "          @�������`  @ ��A���Cp��������\@_\)BL{CbL�                                    Bx�\  T          @�{��{��  >�@ȣ�Cu
��{�N�R@=qB��Cp{                                    Bx�j�  "          @�������Q�>Ǯ@�Cy����Q�@�B
=Ct��                                    Bx�yb  �          @��R�u���H�������C�n�u�~{?�A�  C�H�                                    Bx��  
�          @���L�Ϳ���?���B'p�Cs� �L�Ϳ
=@(�B�.CXxR                                    Bx���  �          @��׼#�
?�33@�33B��fB�W
�#�
@R�\@Z�HB8Q�B�u�                                    Bx��T  T          @�  ��\)?=p�@�p�B��
B�W
��\)@>{@l(�BL33B���                                    Bx���  "          @�p��\)>W
=@��B�.C� �\)@�R@y��BcffB͊=                                    Bx�   S          @��\��G�=�\)@�ffB�\)C/�쿁G�@�@tz�BcffB�                                    Bx��F  �          @�  ����>k�@���B�\)C*ff����@�@eBTz�B�\)                                    Bx���  T          @�{��z�<#�
@���B�Q�C3�=��z�@ ��@_\)BT�Ck�                                    Bx��  �          @�
=��=q��z�@h��B_Q�C^.��=q>\@\)B��3C(^�                                    Bx��8  T          @�\)�k���\)@{�Bx
=Cs��k�>�{@��HB�u�C�q                                    Bx��  "          @�
=���(��@:�HB5ffCv{���Tz�@uB��CWQ�                                    Bx��  �          @��H�p��p  ?�\)A�p�Co�\�p��$z�@C�
B(G�Ce^�                                    Bx�)*  
�          @���G��hQ�?���A�ffCp��G��{@Z�HBB�HCc�                                     Bx�7�  
�          @��H����W
=@
�HA�G�Cs�ÿ�녿�=q@c�
B]��Cd\                                    Bx�Fv  �          @���u�@  @�HB�HC��\�u��33@eB���C�\                                    Bx�U  
�          @��H��  �\)?=p�A��Cyk���  �E@(Q�B  Ct)                                    Bx�c�  �          @�z��{�{�?�RA�Ct���{�E@\)B
��Co                                    Bx�rh  
�          @�(���p��tz�?:�HAG�Cr����p��<��@"�\B=qCl�                                    Bx��  T          @�z��
=�{�>8Q�@�
Cs�)��
=�S�
@
=A�
=Co�\                                   Bx���  �          @�\)��{�>�(�@�
=Cr���L(�@�A��Cl�
                                   Bx��Z  T          @����	���~{=�G�?�
=Cq��	���XQ�@�A�p�Cm��                                    Bx��   �          @��\���~{?
=@�Cq(����H��@\)Bp�Ck.                                    Bx���  �          @��׿޸R���\?O\)A%G�Cv�޸R�HQ�@/\)B�RCp�R                                    Bx��L  T          @�����(���G�?+�A33CtͿ�(��J�H@&ffB�\Cn(�                                    Bx���  �          @�  �G��z�H?aG�A4(�Cr�\�G��=p�@.{B��Ck�q                                    Bx��  "          @�\)���q�?�G�A��\CtY���� ��@L��B7z�Cj5�                                    Bx��>  T          @�ff��
=�:�H@(�B��Cp{��
=���@dz�Bl��CY��                                    Bx� �  �          @�(��B�\�`  ?�33A��
C�⏿B�\��@Y��Bc�Cy��                                    Bx� �  
�          @���333����?�\)Am��C�1�333�B�\@C33B1G�C�|)                                    Bx� "0  
�          @�z�Y�����?�A���C�ÿY���3�
@P��B?�C}&f                                    Bx� 0�  �          @��Ϳ:�H��=q?��RA�{C��H�:�H�1�@Tz�BDz�C0�                                    Bx� ?|  e          @�{�
=q�c�
@�A�ffC��3�
=q�G�@g�Bp
=C�                                    Bx� N"            @���E��^�R@�A��C��f�E����H@e�Bn33Cx�=                                    Bx� \�  T          @�G������b�\?�AӮC}5ÿ����	��@W�BZ=qCs�                                    Bx� kn  T          @z=q��\����@P  Bb�RCV!H��\?
=@Z=qBup�C!��                                    Bx� z  
�          @u��z��%�@G�B��Cqff��zῌ��@P��Bu33CY�3                                    Bx� ��  T          @\)�Tz��P��@   A�=qC��Tz����@W
=Bm(�Cu��                                    Bx� �`  
�          @����aG��,��@5�B3�C|
=�aG��k�@s33B�
=Cbs3                                    Bx� �  T          @x�ÿ����33@Z�HB{z�Cy����>�z�@r�\B��C�                                    Bx� ��  T          @g
=�8Q��ff@]p�B�p�CS�3�8Q�?��@N{B�W
B�ff                                    Bx� �R  
�          @c33����#�
@]p�B��CS�f���?˅@E�By33Bƅ                                    Bx� ��  �          @q녾����@VffB��
C�` ��>�@eB�aHB���                                    Bx� ��  "          @~�R����?aG�@tz�B��\B�녾���@*�H@7
=B:��B�
=                                    Bx� �D  
�          @�=q�����G
=@!�B��C�Z���Ϳ�
=@p  B��fC~h�                                    Bx� ��  �          @��ÿ\(��E�@(�B�C~T{�\(���Q�@j=qB���Co
                                    Bx�!�  �          @�Q�#�
�333@7�B6C�논#�
�}p�@xQ�B�u�C�Ǯ                                    Bx�!6  �          @}p���z��:�H@'�B&�C�%��z῜(�@n�RB��)C�B�                                    Bx�!)�  
�          @�33�z�H�S�
@33A�\)C}��z�H��@[�Bj�
Cr\                                    Bx�!8�  �          @�\)��Q��\��?��RAՙ�Cp�
��Q���R@\��BLz�Ca�f                                    Bx�!G(  �          @�\)�z��X��@G�A��
Cn�{�z��@\��BJ�C^޸                                    Bx�!U�  
�          @����p��g
=?�{A�\)Ctc׿�p��z�@L��B?�\CiB�                                    Bx�!dt  "          @{��O\)�N�R@�
A�\)C���O\)��\@Y��Bp�HCu�=                                    Bx�!s  "          @n�R���H�L��?�\)A�C������H��@N{BmC{                                    Bx�!��  "          @��׿!G��a�@�\BQ�C��Ϳ!G���z�@p  Bw33C{�                                    Bx�!�f  T          @��
�:�H�h��@�A�=qC�T{�:�H� ��@r�\BrQ�Cz
=                                    Bx�!�  T          @��Ϳ�ff�qG�?�p�A�Q�C~c׿�ff��@fffB\�\Cu(�                                    Bx�!��  �          @�(��xQ��o\)?�p�A�33Cs3�xQ��  @eB^��Cv�H                                    Bx�!�X  
�          @������z�H?�ffA�Q�C~s3����'�@S33BE(�Cwff                                    Bx�!��  
�          @�33��=q�o\)?
=q@���Cz^���=q�=p�@B  Cu�=                                    Bx�!٤  �          @���\�xQ�>�=q@n{Cx�f�\�N{@
�HA�\)Ct�\                                    Bx�!�J  T          @�{��{���\?���A`(�C{�Ϳ�{�?\)@>{B(G�Cu�\                                    Bx�!��  "          @�  ��z��{�?��Ay�C}�)��z��5�@=p�B0
=Cw�=                                    Bx�"�  �          @�  ��(��i��?z�HAd��C{���(��*�H@*�HB)p�Cus3                                    Bx�"<  �          @�z��G��n{?xQ�AYCw�q��G��/\)@,��B#�\Cq33                                    Bx�""�  T          @�Q��ff�z=q?�RA{Cx\)��ff�Dz�@\)B�Cs:�                                    Bx�"1�  
�          @~�R���\�`  ?O\)AD  Cz&f���\�(Q�@��B 33CtO\                                    Bx�"@.  �          @|�����U=u?n{Cn
=���7
=?�p�A�  Ci�                                    Bx�"N�  T          @~{�Q��H�þB�\�0��Chٚ�Q��3�
?�33A�
=Ce�=                                    Bx�"]z  
�          @_\)?��
��ff@2�\BY
=C���?��
=L��@L��B�k�@z�                                    Bx�"l   �          @s33?��R��G�@AG�Ba�C�w
?��R?�z�@*=qB=�
Bff                                    Bx�"z�  "          @p��?���
=@1G�BZ��C��{?�?��\@'
=BHA�{                                    Bx�"�l  �          @{�?h���`  ?��RA���C�H�?h����\@B�\BKC��
                                    Bx�"�  �          @i��?����33?�=qBG�C�C�?���#�
@(�Ba=qC��                                    Bx�"��  �          @aG�@ �׾W
=@p�B=�
C��@ ��?k�@G�B)\)A�ff                                    Bx�"�^  
(          @n{?�z�Ǯ@:�HBP=qC�l�?�z�=�G�@S�
B}(�@r�\                                    Bx�"�  �          @h��?��ÿB�\@AG�Bc(�C��q?���?W
=@@  B`�A�
=                                    Bx�"Ҫ  "          @�  @,��>u@P  BHp�@��@,��?�=q@,(�B
=Bp�                                    Bx�"�P  �          @�  @#�
>���@AG�BE��@��H@#�
?��@��BQ�B��                                    Bx�"��  �          @_\)?����33@-p�BZ��C��=?��>\)@C33B�8R@���                                    Bx�"��  
�          @XQ�@
=�W
=@p�B=\)C�)@
=?   @#�
BF�
AUp�                                    Bx�#B  "          @\(�?�\)��z�@p�B9�C�}q?�\)<#�
@5Bbff>B�\                                    Bx�#�  �          @e�?�\)�z�@#�
B7p�C��)?�\)��@P��B��C���                                    Bx�#*�  �          @i��?����
=@)��B9�C��\?��׿�\@VffB���C��q                                    Bx�#94  �          @c33?�  �@(�B.(�C��{?�  �
=@J=qB{�C�8R                                    Bx�#G�  T          @c33?У׿��R@�B-��C��\?У׿�\@FffBtz�C�O\                                    Bx�#V�  �          @c�
?޸R�
=q@�RB(�C�` ?޸R�B�\@@��Bg33C�!H                                    Bx�#e&  T          @[�?Ǯ��?��HB�C�(�?Ǯ��  @5Bc�C��=                                    Bx�#s�  �          @Z�H?�z��G�@  B%�C��{?�z�Y��@E�B��)C��
                                    Bx�#�r  
9          @a�?�G���@�B'�C���?�G��O\)@I��B���C��q                                    Bx�#�  
�          @c33��ff�P��?��\A���C����ff�(�@.�RBJG�C�&f                                    Bx�#��  �          @|�;���s33?h��AU��C�}q����5@+�B+C�S3                                    Bx�#�d  �          @a녾���W
=?@  AJ=qC������"�\@�B)�
C�AH                                    Bx�#�
  �          @P  ?}p��(��?�  A�\)C�>�?}p���ff@'
=B[��C�E                                    Bx�#˰  �          @dz�?���,��?���BQ�C��?����\)@B�\BhG�C��
                                    Bx�#�V  �          @\��?E��0  ?�\A�ffC��{?E���G�@:=qBo(�C��\                                    Bx�#��  �          @��Ϳ�p��r�\?@  A'�Cx����p��:=q@"�\BG�Cr��                                    Bx�#��  �          @�(���z��hQ�>�\)@w�Cr.��z��@��@�\A�\Cm�=                                    Bx�$H  �          @�
=�%��S33��\)�s�
Ch
=�%��@  ?�33A��
CeaH                                    Bx�$�  �          @�\)�>�R�>{��p���G�C`��>�R�0��?�33A}C^�
                                    Bx�$#�  �          @������e����\Cl�3���G�?�  A�  CiB�                                    Bx�$2:  �          @�녿�����H>���@�p�C|p�����XQ�@z�Bp�Cy�                                    Bx�$@�  �          @��
�Ǯ����?L��A'
=Cx�
�Ǯ�Fff@,(�B�CsO\                                    Bx�$O�  �          @�{��
���>�=q@W
=Cpc���
�X��@  A�Q�Ck�                                     Bx�$^,  T          @�Q��   ���H>8Q�@�Cp
=�   �k�@�
AۮCkǮ                                    Bx�$l�  �          @�ff�!���  =�Q�?���Co5��!��h��@(�A�  Ck33                                    Bx�${x  �          @����(���(�=L��?
=Coc��(��c�
@AΣ�Ck��                                    Bx�$�  �          @�ff�Q����H��G���G�Cq@ �Q��tz�@33A�Q�Cn
                                    Bx�$��  u          @��H�
�H���\<#�
>��CsW
�
�H�p  @
=qA�p�Co�                                    Bx�$�j  �          @����ff���ͼ�����Cpk��ff�g�@�Aə�Cl�                                    Bx�$�  �          @��R�����Q�#�
�
=qCn�
����`��?���A��
Ck�                                    Bx�$Ķ  e          @�p�����G�=���?�
=Cp����^{@A�z�Cl#�                                    Bx�$�\  �          @����'��~�R>W
=@�RCl�f�'��W
=@	��A�Q�Ch�                                    Bx�$�  �          @�(�����p  =u?Tz�Cn޸����N�R?�z�A���Cj�                                    Bx�$�  �          @�=q����}p��#�
��CpG�����]p�?�AǅCl�)                                    Bx�$�N  �          @����$z��k���G���z�Ck��$z��Z�H?�
=A�(�Ci                                    Bx�%�  �          @���8Q��\(��O\)�#
=Cf)�8Q��X��?�  AJ�RCe�                                    Bx�%�  �          @�33�5��a녿Y���)�CgE�5��_\)?�G�AI�Cf�                                    Bx�%+@  �          @���(���hQ�J=q� ��Ci�q�(���c33?���A_�
Ci^�                                    Bx�%9�  �          @���  �s�
�0����RCos3�  �j=q?��
A��\Cnh�                                    Bx�%H�  �          @��R�����Q��ff���Ct�����mp�?˅A�\)Cr�q                                    Bx�%W2  �          @��׿�����녾��R�}p�CtaH�����l(�?�p�A�ffCr&f                                    Bx�%e�  �          @��R>#�
�{�?��
A�=qC�(�>#�
�2�\@C33B>�C���                                    Bx�%t~  �          @�{�������
?:�HAQ�C�W
�����\(�@2�\B�C�(�                                    Bx�%�$  �          @�p��������
?��@�{C�Y������a�@(Q�B��C�1�                                    Bx�%��  �          @���>������?�  A�\)C�Q�>���8Q�@Dz�B9��C��)                                    Bx�%�p  �          @�p�?(��j�H?��HA�33C���?(��
=@S33BV33C�@                                     Bx�%�  �          @�?z��l��?�A��HC�u�?z���@Y��B[33C�H                                    Bx�%��  �          @�=q?���G�?�A�{C���?��4z�@N{BA�C�B�                                    Bx�%�b  �          @�  >��R���\?�AqC��>��R�L(�@I��B1��C���                                    Bx�%�  �          @���>W
=��(�?��Ah(�C�\)>W
=�P��@H��B/��C��3                                    Bx�%�  �          @���>k���p�?^�RA2ffC�w
>k��Z�H@;�B"
=C��                                    Bx�%�T  �          @��Ϳ@  ��\)?\(�A5��C��3�@  �P��@5�B C��                                     Bx�&�  �          @�  �W
=���?!G�A��C���W
=�_\)@,(�BffC�B�                                    Bx�&�  �          @�(����H���H>aG�@:=qCzT{���H�]p�@��A��HCw�                                    Bx�&$F  �          @�\)��33�����.{�
=Ct�
��33�g
=?�A�=qCr8R                                    Bx�&2�  �          @���p�������\)�Y��Cqk��p��dz�?�
=A�G�Cn=q                                    Bx�&A�  �          @��
=�l(���z��n�RCmff�
=�Vff?ǮA�Q�Cj�f                                    Bx�&P8  �          @�33�����\��(����HCs�����qG�?У�A�Q�Cq��                                    Bx�&^�  �          @��Ϳ��H���R��33��Q�Ct�q���H�u?�G�A�\)Cr�                                    Bx�&m�  �          @����{��  ��Q����RCvc׿�{�xQ�?�G�A�{Ctp�                                    Bx�&|*  �          @��׿�{��(������=qCw
=��{�x��@ ��AǙ�Ct��                                    Bx�&��  �          @��Ϳ������>aG�@0��Cz:�����l��@�A��Cw                                    Bx�&�v  �          @�p���(����>�z�@c33C{�{��(��l��@��A��RCxff                                    Bx�&�  �          @�p���
��p�>�=q@S�
Cs�3��
�`��@G�A�33Co��                                    Bx�&��  �          @�33�p��u�   �ȣ�CmaH�p��fff?�Q�A��Ck�f                                    Bx�&�h  �          @�33�����
�u�G�C{�ÿ��u@z�AָRCy��                                    Bx�&�  �          @��R����(�?(�@�z�C�����a�@*�HBffC�Q�                                    Bx�&�  �          @���+����?�@�p�C����+��_\)@&ffB��C���                                    Bx�&�Z  �          @��>�z�����?���Af{C���>�z��Fff@<��B-��C���                                    Bx�'    �          @��\>��xQ�?�=qA�  C�` >��0  @C33B>G�C���                                    Bx�'�  �          @�Q�����z�?z�HAH��C�
=����W
=@?\)B&z�C���                                    Bx�'L  �          @�=q>�  ���?�\)A�
=C�˅>�  �8Q�@J�HB>p�C��                                    Bx�'+�  T          @���?   ����?�\)A���C���?   �7
=@I��B=p�C�H                                    Bx�':�  �          @|��@1��G�?�AǮC�aH@1녿�@#�
B!\)C���                                    Bx�'I>  �          @|(�@/\)�   ?�33A�z�C���@/\)�Tz�@(Q�B*��C���                                    Bx�'W�  �          @|(�@C33���\@G�A��C��=@C33��@Q�B  C��)                                    Bx�'f�  �          @~{@Fff��
=?�Q�A��C��)@Fff���R@��B{C�"�                                    Bx�'u0  �          @��@S�
���@�\A��
C���@S�
�#�
@B�RC���                                    Bx�'��  �          @�=q@\(��J=q@   A��
C���@\(�>�=q@Q�A�p�@���                                    Bx�'�|  �          @���@e��   ?�A�ffC�\@e�>�G�?���A�=q@�                                      Bx�'�"  �          @xQ�@B�\�:�H@\)B�HC�G�@B�\>�ff@z�B(�A�H                                    Bx�'��  �          @xQ�@G
=�.{@G�B=qC�ff@G
=?p��@z�B  A��H                                    Bx�'�n  T          @xQ�@Vff�#�
?���A�=qC��q@Vff=�G�?\A¸R?�                                    Bx�'�  �          @s33@X�ÿz�H?�{A���C���@X�þW
=?�A�
=C�C�                                    Bx�'ۺ  �          @u@^{���?��RA�(�C�P�@^{��Q�?�{A�C�3                                    Bx�'�`  �          @n�R@Vff�W
=?���A�(�C��@Vff���
?�\)A��C�\)                                    Bx�'�  �          @s33@XQ�5?ǮA�{C�f@XQ�=�?��HA�z�@z�                                    Bx�(�  �          @xQ�@_\)�L��?��RA�=qC��=@_\)<#�
?�Q�A�ff>W
=                                    Bx�(R  �          @w
=@\(��\(�?\A�(�C��q@\(����
?�  A׮C��{                                    Bx�($�  T          @qG�@[��}p�?���A��RC��@[�����?�(�A��HC�E                                    Bx�(3�  �          @���@b�\���?\A�(�C�3@b�\��=q?��Aߙ�C��{                                    Bx�(BD  �          @�(�@j=q�J=q?�p�A�=qC��@j=q>\)?��AڸR@�                                    Bx�(P�  �          @�Q�@b�\����?��
A�=qC�Ф@b�\?�\?�  AУ�A{                                    Bx�(_�  T          @}p�@o\)���?��\A���C���@o\)>��?�Q�A�@�                                    Bx�(n6  �          @�G�@k�>��
?У�A��R@���@k�?��?��
A�{A�Q�                                    Bx�(|�  �          @�  @b�\?��?�  AϮA�R@b�\?�\)?�  A��A���                                    Bx�(��  �          @tz�@Vff?��?�
=A��RA��
@Vff?�{?@  A6{A��                                    Bx�(�(  �          @i��@:=q�L��@	��B�C�w
@:=q?z�H?�A�A���                                    Bx�(��  �          @�G�@:�H��@<��B3�
C���@:�H?�G�@3�
B)33A�Q�                                    Bx�(�t  �          @�(�@J�H���
@VffB9��C�'�@J�H?���@EB(  A���                                    Bx�(�  e          @�G�@P  �\)@I��B0\)C���@P  ?���@5B33A���                                    Bx�(��  
�          @�  @R�\>W
=@B�\B*p�@l(�@R�\?У�@$z�BA���                                    Bx�(�f  �          @�
=@G��&ff@G�B1�
C�  @G�?k�@C�
B-  A�
=                                    Bx�(�  
�          @��
@C33�\(�@?\)B-ffC��@C33?(��@B�\B1{ADQ�                                    Bx�) �  �          @�=q@W�>�@*�HBp�@
=q@W�?���@�\B ��A�(�                                    Bx�)X  �          @s33@Q쿣�
@!�B,C�ٚ@Q�=��
@5BH(�?���                                    Bx�)�  �          @}p�?˅�0  @
=Bp�C��?˅��ff@XQ�Bj�
C�^�                                    Bx�),�  �          @�33?�=q�AG�@!G�BC��R?�=q��(�@i��Bd��C��q                                    Bx�);J  
�          @w
=@z��33@��B(�C�U�@z�
=q@@��BNQ�C�j=                                    Bx�)I�  �          @q�@z��z�@�
B��C�u�@z�:�H@@��BWC�9�                                    Bx�)X�  "          @��R?���N{@��B	��C��?�����H@hQ�Bm33C���                                    Bx�)g<  "          @�\)?�=q�W�@.{B\)C��f?�=q��
=@\)B}�\C�^�                                    Bx�)u�  T          @�
=?p���g
=@p�B�HC�E?p���G�@xQ�Bp��C�p�                                    Bx�)��  �          @��R��\)��Q�@=qA�Q�C�|)��\)���@���Bl�C�"�                                    Bx�)�.  �          @�  ���
���@33A��C�޸���
�%@\)Bd
=C��=                                    Bx�)��  
�          @�  �����\)@
=qA׮C��)����.{@y��B\�C�k�                                    Bx�)�z  �          @�=�G���  ?���A���C�� =�G��5�@n{BR��C��                                    Bx�)�   T          @��<��
��Q�?�p�A��HC�*=<��
�<��@a�BH��C�=q                                    Bx�)��  
�          @��\>B�\��(�?�p�Az�RC�9�>B�\�R�\@H��B.��C��H                                    Bx�)�l  �          @�ff�u���H?O\)A(��C���u�\��@0  B33C��H                                    Bx�)�  �          @�
=������
?E�A{C�Q����`  @.�RB��C��=                                    Bx�)��  �          @��R�aG����?h��A<  C��{�aG��[�@6ffB��C�1�                                    Bx�*^  �          @��;�z���Q�?�ffAQp�C�/\��z��`  @B�\B#�RC���                                    Bx�*  
�          @�z������@
=A�  C�����*�H@��\Baz�C��                                    Bx�*%�  "          @�G���z���z�@�A�C�þ�z��/\)@�ffBb�C���                                    Bx�*4P  �          @�����z����@�\A��C�!H��z��9��@��
BZ��C�)                                    Bx�*B�  
�          @���aG���
=@Q�A��C�}q�aG��(Q�@�=qBdG�C��3                                    Bx�*Q�  �          @�Q�B�\��\)@{A��HC��H�B�\�;�@���BXffC�R                                    Bx�*`B  �          @��;�ff��
=?��A���C�#׾�ff�E�@p��BI�\C���                                    Bx�*n�  �          @��������?�=qAV�RC�b�����Y��@@��B ��C}�                                    Bx�*}�  
�          @��R�˅����>�(�@���Cz!H�˅�l(�@�HA���Cv�R                                    Bx�*�4  �          @��
�У�����>�p�@��
CyJ=�У��h��@�A�=qCu��                                    Bx�*��  �          @�
=����ff>\@��RC{B����b�\@�\A��Cx
                                    Bx�*��  �          @�\)�����  >�G�@�{C}�����c�
@
=A���Cz
                                    Bx�*�&  T          @�����
=����?�p�Az�RCwaH��
=�@��@=p�B"p�Cp��                                    Bx�*��  
�          @��\����?��HAu��C{5ÿ��I��@@��B$G�Cu��                                    Bx�*�r  
Z          @�33�\�~�R?Y��A5G�Cy#׿\�HQ�@&ffB��Ct.                                    Bx�*�  T          @�����H�|��?^�RA7\)Cv�����H�Fff@&ffB�Cq5�                                    Bx�*�  �          @�p����s33?��A��Ct� ���0  @<(�B'33Cl��                                    Bx�+d  
�          @�\)�ff�p  ?�  A�  Cp�3�ff�0  @5BffCh�{                                    Bx�+
  �          @�{��
=�|��?
=q@�G�Cs����
=�P��@33A�=qCoJ=                                    Bx�+�  "          @�p���=q�\)>�z�@s33Cu\)��=q�Z=q@A�p�Cq�                                    Bx�+-V  �          @�33�������\��ff���C{zΌ������?˅A�Q�CzO\                                    Bx�+;�            @�(���p����ý�����Cx���p��w
=?�{A�{Cu�)                                    Bx�+J�  T          @��
��33��G���33���
Cy  ��33�~�R?�z�A�Q�Cw}q                                    Bx�+YH  
�          @����(���ff�333��Cw���(����?�ffA�G�Cw                                    Bx�+g�  T          @�33��33����(���G�Cu(���33�}p�?��A��CtT{                                    Bx�+v�  �          @��H�1��g�=L��?(�Chu��1��L��?ٙ�A��RCd��                                    Bx�+�:  "          @��*�H�u=�\)?Q�Ck+��*�H�XQ�?�A�z�Cg�R                                    Bx�+��  �          @����0  �x�þu�:=qCj��0  �dz�?ǮA�z�ChW
                                    Bx�+��  
(          @�p���\�x�þ�p���(�CrB���\�hQ�?�Q�A�Q�Cp��                                    Bx�+�,            @�������s33�@  �G�Co8R����n{?��A[�
Cn�3                                    Bx�+��  �          @��
�G��p  ��z�����CnǮ�G��~�R>�ff@�=qCpJ=                                    Bx�+�x  "          @���
�~{�:�H��
Co�
��
�w�?�z�Af�\Co.                                    Bx�+�  �          @��������33?#�
@���Co������W�@�A�Cj��                                    Bx�+��  �          @�=q��
��?&ff@��RCq
=��
�[�@{A��RCl�                                    Bx�+�j  T          @�=q�   ����?5Az�CnY��   �R�\@{A��Ch�=                                    Bx�,	  �          @���1��vff?@  AQ�Cj.�1��E@=qA�
=Cd\                                    Bx�,�  "          @�G��5�qG�?:�HAG�Ci
=�5�A�@
=A�RCb�f                                    Bx�,&\  �          @�G��C�
�fff?J=qA��Ce���C�
�6ff@�A��C^�R                                    Bx�,5  �          @����C�
�hQ�?(��@��Ce޸�C�
�<(�@{A�\)C_�H                                    Bx�,C�  �          @����5�qG�?5A��Ci{�5�C33@�A�(�Cc{                                    Bx�,RN  �          @�ff�8Q��j=q?&ff@�\)Cg�\�8Q��>�R@�RA�z�Ca�                                    Bx�,`�  �          @����(Q��z�H?�R@���Cl��(Q��N{@z�A���Cf�3                                    Bx�,o�  "          @����%���Q�>���@dz�CmJ=�%��\��@z�A�
=CiB�                                    Bx�,~@  "          @�  �,���w
=?(�@��
Ck{�,���K�@�A��Ce�3                                    Bx�,��  �          @�  �1G��u>�\)@UCj��1G��S�
?�(�A��
Ce�q                                    Bx�,��  T          @�ff�2�\�o\)>�@��CiO\�2�\�HQ�@
=A�p�CdO\                                    Bx�,�2  �          @�{�*�H�u>�  @G
=Ck=q�*�H�U�?���A�(�CgB�                                    Bx�,��  �          @���&ff�w
==�Q�?�33Cl
=�&ff�Z=q?�A��RCh�3                                    Bx�,�~  �          @��333�n{����33Ci��333�a�?�  Aw�
Cg��                                    Bx�,�$  T          @��H�9���^{�
=��Cf&f�9���W�?�G�AMp�CeW
                                    Bx�,��  e          @����\(��N{�333�(�C_{�\(��L(�?L��A�RC^�H                                    Bx�,�p  C          @�
=�@���dz���H��  Ce���@���Z�H?�33AbffCd��                                    Bx�-  "          @��5�i���&ff���RCh!H�5�c�
?��AM�Cgk�                                    Bx�-�  �          @����9���\�Ϳ���
=Ce�R�9���Vff?��\AO
=Ce�                                    Bx�-b  �          @���4z��c�
�5�{Cg�{�4z��`  ?p��A<��Cg(�                                    Bx�-.  �          @�p��C�
�aG��\)�ٙ�Ce��C�
�N{?�
=A�
=Cbz�                                    Bx�-<�  �          @����R�\�C33�z���p�C^���R�\�@  ?Q�A'�C^L�                                    Bx�-KT  �          @�=q�C33�W
=����=qCc�q�C33�N{?���AX(�Cb�                                    Bx�-Y�  �          @�{�*=q�[��aG��7
=Ch0��*=q�]p�?8Q�A=qChz�                                    Bx�-h�  T          @����(��\�Ϳ�
=��p�Cm���(��g�>�ff@��
Cnٚ                                    Bx�-wF  �          @�  ��G��XQ��{��z�Cr����G��w
=�B�\�#33Cu}q                                    Bx�-��  �          @�{�   �g
=�Ǯ��\)Cq  �   �z�H>8Q�@��Cs                                      Bx�-��  �          @�
=�ٙ����ÿ=p��ffCxQ�ٙ���p�?��HApQ�Cw�\                                    Bx�-�8  "          @�Q���H��p���(���{Cx�׿��H���?�ffA��Cw��                                    Bx�-��  
�          @��׿������
�\���Cw!H������33?���A�=qCu�                                    Bx�-��  �          @���������;u�8Q�Cv�=������?ٙ�A���Cu�                                    Bx�-�*  �          @��\������u�333Ct��������?�Q�A��Cr��                                    Bx�-��  �          @��H�����
���Ϳ�Q�Ct�����~�R?�A��Crc�                                    Bx�-�v  �          @��H�
�H��=q���Ϳ�33CsaH�
�H�{�?��A�(�Cq(�                                    Bx�-�  �          @��H��R���þL����Crz���R�|(�?�
=A��Cp}q                                    Bx�.	�  �          @�z������ͽ#�
���HCtY����\)?�\)A�ffCr�                                    Bx�.h  T          @�z��(����
���
�fffCsn�(��~{?���A�\)Cq.                                    Bx�.'  "          @�z�����p������
Ct�
����G�?�ffA��\Cr�\                                    Bx�.5�  !          @��
�Q���33��
=��ffCs��Q����?�  A�  Cr�{                                    Bx�.DZ  �          @�z��������?0��A(�Cp
����\(�@�HA��CkE                                    Bx�.S   
�          @����
���\?�  Ao
=Cp� ��
�H��@8��B{Ci�3                                    Bx�.a�  4          @�z���H��(�?xQ�A7�
Co����H�S�
@*=qB��Ciٚ                                    Bx�.pL  �          @�(��z���
=?�R@��
Cq(��z��b�\@Q�A�ffCl��                                    Bx�.~�  �          @��{���?s33A1�CoY��{�Vff@)��B��Ci�f                                    Bx�.��  �          @���
��ff?�33AX��Cq.��
�S33@5B�Cj�q                                    Bx�.�>  �          @�
=�z����H?
=q@�Q�Cq��z��l(�@ffA�ffCm�f                                    Bx�.��  �          @�{�"�\��
=����Q�Cn���"�\�w�?�Q�A��RCl��                                    Bx�.��  
�          @�p��*=q�����
=���RCm��*=q�x��?�\)A��Ck�f                                    Bx�.�0  
Z          @���!G����R��Q쿅�Co��!G��vff?�(�A���Cl�                                    Bx�.��  T          @�������  �#�
���CpE���z�H?�A�
=Cn0�                                    Bx�.�|  �          @��
��R���׿z����
Crn��R���?��Av�HCq��                                    Bx�.�"  T          @���G���  �!G���Q�Cqٚ�G����
?�p�Aj�\Cq
                                    Bx�/�  �          @��
���\)����\)Cp�������?�\)A�\)Co�                                    Bx�/n  T          @��H��R����>#�
?�{Crp���R�s�
?��HA���Co�H                                    Bx�/   �          @��
��
��G�=L��?z�Cq�3��
�w�?�{A�p�Co#�                                    Bx�/.�  �          @�p������=u?5Crff���{�?�33A�\)Co޸                                    Bx�/=`  "          @����\)�������p�Cr�
�\)�}p�?�A�\)Cp�{                                    Bx�/L  T          @����R��(��L�Ϳz�Cr����R�\)?�ffA��Cp                                    Bx�/Z�  �          @���\)���
����p�Cr�)�\)����?�(�A�(�Cp�)                                    Bx�/iR  
�          @�p��
=���������x��Cq=q�
=��G�?�  A�ffCo��                                    Bx�/w�  �          @�p�������þ��R�e�Cp� �����Q�?\A��Co(�                                    Bx�/��  �          @�z��(�����׿G���\Cl�f�(���~{?z�HA9p�Cl\)                                    Bx�/�D  �          @����%���33�(�����Cmٚ�%���  ?�\)AS33Cm5�                                    Bx�/��  �          @�z��{���\��33���HCr���{���\?��RA�z�Cqz�                                    Bx�/��  �          @�z��  ��녿���\)CraH�  ��z�?��AyG�Cqk�                                    Bx�/�6  T          @����H���Ϳ:�H�	��Co�=��H���\?���AK33CoW
                                    Bx�/��  �          @��
�p��������θRCoY��p���Q�?�p�Aj�\Cnn                                    Bx�/ނ  �          @��
�$z���녿E����Cm�f�$z�����?}p�A:�RCmY�                                    Bx�/�(  �          @��
�����p��z����Co��������?���Ad��Cn��                                    Bx�/��  �          @���'���=q�=p���Cm:��'���Q�?�G�A=�Cl�)                                    Bx�0
t  �          @���:=q�w
=�W
=�G�Ci  �:=q�w
=?Tz�AQ�Ci�                                    Bx�0  T          @����+����׿:�H�	�ClW
�+��~{?}p�A;\)Ck�R                                    Bx�0'�  �          @�z��"�\���
�5��CnY��"�\��G�?�ffAF�HCm�H                                    Bx�06f  
�          @���4z��~{������Cj���4z��u?���AV{Ci�R                                    Bx�0E  B          @��H��R�����G���
=Cn���R�{�?��Ax(�CmǮ                                    Bx�0S�  "          @����  ��
=��33��{Cr  �  ��  ?�z�A�
=Cp��                                    Bx�0bX  
�          @����   �|�Ϳ\(��&�RCm���   �}p�?Tz�A!p�Cm��                                    Bx�0p�  �          @���������?�R@�\Cv�Ϳ���qG�@
=A�Q�Csc�                                    Bx�0�  �          @�(���G���
=?W
=ACx���G��o\)@%B 33Ct�\                                    Bx�0�J  �          @�(���\)��>���@���Cw)��\)�xQ�@
�HA�{CtE                                    Bx�0��  �          @��\����z�?��AP��Cy����dz�@1�B�Ctٚ                                    Bx�0��  �          @��\�ٙ�����?�Q�A���CxB��ٙ��S�
@C33BffCr�\                                    Bx�0�<  �          @�������q�@'�B�Cz��������@z�HB\G�Cp�{                                    Bx�0��  �          @�\)�B�\�{@x��Bf�C��f�B�\�.{@���B�W
C~B�                                    Bx�0׈  �          @�G��Q���Q�@{A���C�<)�Q��*�H@w�BY
=C|��                                    Bx�0�.  �          @�=q��Q���G�@(�A�ffCzh���Q��3�
@hQ�BD  Cr�                                    Bx�0��  �          @����=q��z�?�G�A�Cv!H��=q�E�@Q�B)CoB�                                    Bx�1z  �          @��
������R?�@�p�Cpk�����g�@p�A��Cl�)                                    Bx�1   �          @�z��$z���G�?xQ�A733Cm�\�$z��S�
@ ��A��Ch33                                    Bx�1 �  �          @����<���w
=?�@�Ch�f�<���S�
@z�A�Q�CdO\                                    Bx�1/l  �          @���   ���
?��A�=qCt.�   �J=q@C�
B{Cm                                    Bx�1>  �          @����\��
=?���A�Q�Cw=q��\�N�R@J=qB"p�Cq@                                     Bx�1L�  �          @�(���(����?�=qA���Ct�׿�(��K�@G
=B�
Cn8R                                    Bx�1[^  
�          @����=p��x��>���@_\)Ch���=p��\(�?���A��HCeQ�                                    Bx�1j  
�          @�(��333�~�R>�z�@W
=Cj�{�333�b�\?���A��Cg�
                                    Bx�1x�  �          @�(��-p��\)?(��@���Ck�
�-p��Y��@(�A��
CgxR                                    Bx�1�P  
�          @�z���H���?��AD(�Co����H�W
=@%�A�p�CjE                                    Bx�1��  �          @���	�����?h��A*�HCs� �	���fff@"�\A��
Co(�                                    Bx�1��  �          @�p��G���{?(�@�z�Cu���G��u@z�A���Cr@                                     Bx�1�B  �          @�(���
���?B�\A33Cp����
�^�R@z�A���Cl\)                                    Bx�1��  
�          @��\����z�>�@�Q�C�׿���=q@G�A�C~�                                    Bx�1Ў  �          @��������G�?�@ۅC}�׿���|��@�
A�{C{��                                    Bx�1�4  T          @��H��(���Q�        Cy.��(�����?�\A�=qCw��                                    Bx�1��  �          @�(���(���=q?5A33C|(���(��z�H@(�A�=qCyu�                                    Bx�1��  �          @�p��	�����H?5A=qCs�f�	���mp�@ffA�{Co�                                    Bx�2&  �          @��Ϳ�z���\)>Ǯ@��
Cvٚ��z��~{@
=A��HCt@                                     Bx�2�  �          @��H��{��{�\)�У�C�W
��{���
?ٙ�A�p�C�                                     Bx�2(r  �          @��\�����������C�>��������?�(�A��HC�                                    Bx�27  �          @������H��
=����33Cy
���H��(�?��HA��Cw��                                    Bx�2E�  �          @�\)�n{���Ϳ#�
�{C���n{���?�\)Ac\)C��f                                    Bx�2Td  �          @���?����=q�����G�C�ٚ?������#�
�33C��                                     Bx�2c
  �          @�������  ��(���G�C��R������\=u?8Q�C�3                                    Bx�2q�  �          @�G��L����\)��G����C���L����\)>u@@��C���                                    Bx�2�V  �          @�p��Ǯ��z�G��!G�C�y��Ǯ��33?xQ�AG�
C�s3                                    Bx�2��  
�          @�Q�n{���
�����\C�P��n{��{?���A��RC�R                                    Bx�2��  T          @��׿������>�\)@X��C:Ῑ����33@ ��A�\)C}��                                    Bx�2�H  T          @�
=�#�
��{�aG��,��C�f�#�
��p�?˅A���C��
                                    Bx�2��  �          @�
==�����z�8Q��C��
=������?�{AZ=qC��R                                    Bx�2ɔ  �          @�z�>�33��zῪ=q��Q�C�E>�33���>�ff@�33C�/\                                    Bx�2�:  �          @���>��y���{��33C��H>���ff�\)��(�C�{                                    Bx�2��  �          @���>\��G���\�ծC���>\���׾�33��C�e                                    Bx�2��  �          @�ff>\)��33��\���RC���>\)��{<��
>k�C��R                                    Bx�3,  �          @�  >�������ff�~{C��R>����R?�@���C���                                    Bx�3�  �          @��?������(����C�aH?����G�?���Aj�HC�w
                                    Bx�3!x  �          @�Q�?���ff�   ����C�%?���G�?��A�Q�C�AH                                    Bx�30  �          @�Q�=�Q��������{C���=�Q�����?�G�A��\C��                                    Bx�3>�  T          @�33?xQ��C�
�!G��1��C��H?xQ��Dz�?�A Q�C��R                                    Bx�3Mj  �          @�z��C33�~�R?�\)AH��Ch��C33�QG�@!�A�  Cb�R                                    Bx�3\  �          @��
�<(����
?.{@�p�Cj���<(��dz�@�A��Cf}q                                    Bx�3j�  �          @�ff��\)�hQ쿙����\)Cr���\)�s�
>u@Q�Cs�                                    Bx�3y\  �          @��H�޸R��{�
=q��33Cw� �޸R���H?��AV�HCw�                                    Bx�3�  �          @����'����\�����H  CmE�'���?�@�=qCm�                                    Bx�3��  �          @�G��K��a녿�z����Cc�q�K��s33<#�
=�G�Cf{                                    Bx�3�N  �          @��\�{��  ?�z�A�Q�Crz��{�U�@G�B�ClaH                                    Bx�3��  �          @��\������?�Q�A�Cp�)����Y��@:=qB�Cj�                                    Bx�3  T          @��H�5�y��>��?�ffCi�f�5�dz�?�=qA�G�Cgn                                    Bx�3�@  �          @�33�AG����H<�>���Ci�=�AG��r�\?��
A��RCgp�                                    Bx�3��  �          @�p���Q���33?�=qAn�RCw)��Q��r�\@<(�B
ffCr�{                                    Bx�3�  �          @���
=��p�?�z�A}p�Cz@ ��
=�u�@A�B�
CvQ�                                    Bx�3�2  �          @��Ϳ�\)����?�G�A�  Cz� ��\)�qG�@G
=B�\Cv�3                                    Bx�4�  
�          @����{��=q?ǮA���Cz�Ϳ�{�k�@HQ�B�\CvQ�                                    Bx�4~  �          @��׿����?�{A{�Cw
=����hQ�@8��B�Cr��                                    Bx�4)$  T          @�����(����R?�G�AhQ�Cv.��(��l(�@3�
BffCq�                                    Bx�47�  �          @��H�z����?�  A��CtL��z��W�@L��B��Cnh�                                    Bx�4Fp  �          @�z��
=q���\?��HA�(�Css3�
=q�Z=q@J�HB�RCm��                                    Bx�4U  �          @�33��p���z�?�z�A�  Cu�3��p��_\)@H��B33CpaH                                    Bx�4c�  �          @�33��(���ff?��
A�{Cv
��(��e@B�\Bz�Cq33                                    Bx�4rb  �          @��H��(���\)?���Av�\CvE��(��l(�@8Q�B
33Cq�f                                    Bx�4�  �          @�33��
��  ?���AYG�Cuk���
�p��@/\)BG�CqT{                                    Bx�4��  �          @��
�����?��Am��Cw&f���q�@7�BQ�Cs�                                    Bx�4�T  �          @���33��Q�?���A�z�Cu�{�33�l(�@>�RB�
Cp�3                                    Bx�4��  �          @�p��z�����?��Ay�Cup��z��n�R@:�HB	�Cp��                                    Bx�4��  T          @��������?��
Af=qCt&f���n�R@3�
B  Co�                                    Bx�4�F  �          @�{�ff��p�?��Ao�Cr
=�ff�i��@5�B��CmB�                                    Bx�4��  �          @��R�p�����?�=qAl��Cs���p��p  @7
=B33Co��                                    Bx�4�  �          @��"�\����?�z�A}�Co=q�"�\�_\)@5Bz�Ci��                                    Bx�4�8  �          @�ff�:�H�z=q?�\A��Ci:��:�H�A�@B�\Bz�Cb                                      Bx�5�  T          @��A��e�@�RA�G�Ce�R�A��#33@VffB �C\)                                    Bx�5�  �          @�\)�2�\�xQ�@
=A�Q�Cj5��2�\�8Q�@UB=qCaٚ                                    Bx�5"*  �          @�  � �����H@A�Q�Cnp�� ���E@Y��B"�Cfٚ                                    Bx�50�  �          @�\)�!G���z�?��HA�  Cn���!G��K�@R�\B(�Cg��                                    Bx�5?v  �          @����=q���@ ��A�\)Cpff�=q�P��@XQ�B33Ci�=                                    Bx�5N  �          @�Q��(���{@�A�  Co��(��L��@W�B��Ch�R                                    Bx�5\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�5kh   �          @�33�7
=�1G�@J=qB�C`��7
=���R@{�BJz�CO�                                    Bx�5z  �          @����\)�h��@'
=A��RCk���\)�   @n{B9�Ca�                                    Bx�5��  �          @��\�
=����?�{A�G�Cpn�
=�U�@=p�B�
Cj�R                                    Bx�5�Z  �          @���
=��33?aG�AG�CuW
�
=��Q�@�HA��Cr5�                                    Bx�5�   �          @�=q��
=���?��
A<��Cw
=��
=�z�H@#33A�\Cs�                                     Bx�5��  �          @�=q��G���(�?uA.�RCy.��G���Q�@   A�CvL�                                    Bx�5�L  �          @��\�Ǯ����?
=@ָRC{�׿Ǯ����@p�A��HCy�                                    Bx�5��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�5��   �          @�z��3�
�j=q@G�A�(�Chz��3�
�)��@Y��B%\)C_T{                                    Bx�5�>  �          @����&ff��z�?���A��RCm��&ff�U@:�HB
�\Ch#�                                    Bx�5��  �          @�=q����
=?���AD��Cu�\���u�@"�\A���Cr\                                    Bx�6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�60   �          @�������@\)A��Co����AG�@^�RB(�Cg�)                                    Bx�6)�  T          @����\����@G�A���Cpc���\�@��@`  B+Q�Ch�                                    Bx�68|  �          @����33��
=?�A��Cqh��33�Vff@H��B�Ck�=                                    Bx�6G"  �          @�(���(��l(�@;�B{Cq�f��(��{@�Q�BN�Cgz�                                    Bx�6U�  �          @�z῰���S�
@j=qB6ffCwk����׿�\)@�=qB|{Ci�3                                    Bx�6dn  �          @�p����H�H��@S33B&�RCm����H��@��Bd�\C_!H                                    Bx�6s  �          @��{��=q@��BS�\CX��{��33@�{Br�C<{                                    Bx�6��  �          @��
�"�\��ff@�p�B^p�CO�"�\>8Q�@��Bn��C/޸                                    Bx�6�`  �          @�G������`��@)��B��Cp������H@k�BG\)Cg�                                    Bx�6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�6��   �          @��H��p���
=@��A���Cw���p��N{@b�\B0ffCq�R                                    Bx�6�R  �          @��H�޸R��{@�\A�=qCw� �޸R�L(�@c�
B1��Cqh�                                    Bx�6��  �          @��\��33��  @�RAУ�Cx���33�P��@aG�B/��Cs(�                                    Bx�6ٞ  �          @�녿�
=���?�{A��Cx녿�
=�_\)@L��BQ�CtO\                                    Bx�6�D  �          @�=q��z���Q�?��Ak�C�,Ϳ�z����@2�\B�C~�                                    Bx�6��  �          @��\��G����?�ffAm�C�{��G���33@4z�B��C��                                    Bx�7�  �          @�녿Y�����?��
Ak�
C����Y�����@333B��C�+�                                    Bx�76              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�7"�   �          @��>�  ���H��33����C�s3>�  ��{?�G�Ap��C��                                     Bx�71�  �          @���>�  ��(���Q����RC�}q>�  ��\)?�G�An�HC��=                                    Bx�7@(  �          @�(�>�{��������z�HC��q>�{��ff?��
Ar�RC�                                    Bx�7N�  �          @�=q>#�
��녾�z��\��C���>#�
��z�?�ffAz{C���                                    Bx�7]t  �          @��
>\��33�����(�C�9�>\��(�?���A�(�C�T{                                    Bx�7l  �          @��>.{���\������C��>.{��ff?�Q�Ab�HC�
=                                    Bx�7z�  �          @�(�������=q�333���C�c׽�����G�?fffA*�RC�c�                                    Bx�7�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�7�   �          @�zᾳ33���
��=q�L��C�녾�33��{?���Ay�C��R                                    Bx�7��  T          @�(��\��33�k��,(�C��=�\��p�?���A��RC��3                                    Bx�7�X  �          @�z�
=q���H��33���\C��
�
=q��{?�p�Ai��C���                                    Bx�7��  �          @���5��33�#�
��C�ٚ�5����?�33A��C���                                    Bx�7Ҥ  �          @�(��
=q���\����ffC��\�
=q��
=?�{ARffC��q                                    Bx�7�J  �          @��;\���
��p���G�C��f�\��\)?��HAe�C��{                                    Bx�7��  T          @�{�����ͽu�0��C�O\�����?�G�A���C�,�                                    Bx�7��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�8<   �          @��>��������ff�MG�C��>�����ff>��@��C�f                                    Bx�8�  �          @�
=>���z����(�C��
>����?��\AHQ�C�f                                    Bx�8*�  �          @�33�   ��Q�?   @�ffC�H�   ���?�p�A�ffC���                                    Bx�89.  �          @�(��\)���?
=q@��HC��=�\)��z�@�A�
=C�Z�                                    Bx�8G�  
�          @��
�����G�?!G�@�p�C��;����33@
=A��
C�N                                    Bx�8Vz  �          @�������\?Y��A�HC��R�����=q@�AᙚC�O\                                    Bx�8e   �          @�ff������G�?�33AV�\C��)������@%A��C�E                                    Bx�8s�  �          @�ff�\��=q?���AF�HC�� �\���@!G�A���C�q�                                    Bx�8�l  �          @���z�����?�A[
=C�Ff��z���@&ffA��HC�f                                    Bx�8�  �          @���\)��\)?�{A�C�T{��\)��=q@0��BffC�\                                    Bx�8��  T          @��R�\��  ?���A|z�C��þ\���H@0��B\)C�XR                                    Bx�8�^  T          @�\)������?�Q�A\��C��=�����ff@'�A��\C�0�                                    Bx�8�  �          @�  �����H?�{AM��C�޸����Q�@#�
A�=qC�q�                                    Bx�8˪  �          @��׿����H?�
=AZ�RC��{�����@'�A�z�C���                                    Bx�8�P  T          @�Q��ff��(�?�  A7�C�b���ff���H@��A���C�\                                    Bx�8��  �          @�
=�u��z�?G�AG�C���u���@  A��
C�k�                                    Bx�8��  �          @�?z���zἣ�
���C�\)?z����?�p�A�  C��f                                    Bx�9B  T          @��R�L�����?
=@�(�C���L����  @z�AŮC���                                    Bx�9�  �          @�G�������?�ffAA��C�XR�����H@   A�\C�H                                    Bx�9#�  �          @�G���G���z�?���AO33C�ff��G����@#�
A�
=C�                                    Bx�924  �          @����\)��{?Tz�A�C��)�\)��ff@�\A؏\C�b�                                    Bx�9@�  �          @�녿333��
=?&ff@��
C����333����@Q�A���C���                                    Bx�9O�  �          @��þ��H��p�?h��A&=qC�&f���H���@ffA�p�C��3                                    Bx�9^&  �          @�  ��
=���
?�  A9C��=��
=���H@�A��C�=q                                    Bx�9l�  �          @�Q쾸Q����
?�\)AO�C��ᾸQ�����@"�\A��
C��3                                    Bx�9{r  T          @��ÿ�����?�AV�RC��H�������@%�A��\C�Q�                                    Bx�9�  �          @�Q�����33?�ffAA�C�y�������@p�A�C��                                    Bx�9��  �          @�ff�
=����?L��A�HC�|)�
=��=q@��A��
C�q                                    Bx�9�d  �          @�z��
=���B�\�33Cv� ��
=���R?��@���Cv�)                                    Bx�9�
  �          @�p���
���
�h���)CtǮ��
��{>�G�@�(�Cu#�                                    Bx�9İ  �          @�p���\)��\)�333��\CwO\��\)��\)?+�@��HCwT{                                    Bx�9�V  �          @�p���33���H����z�Cz33��33��G�?\(�A"{Cz                                      Bx�9��  �          @���z�H���׼#�
��C�4{�z�H���?�33A��C���                                    Bx�9�  �          @�ff�p����
==�\)?O\)C�XR�p�����?��HA�G�C�\                                    Bx�9�H  T          @����  ��Q�?(�@�ffC�3��  ���
?��RA�C��\                                    Bx�:�  �          @��׿�33����?.{@�Q�C}�f��33���@33A��C|33                                    Bx�:�  �          @�\)�������R>k�@(��C{@ ������?���A�(�Cz+�                                    Bx�:+:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�:9�   �          @�녿����?xQ�A0(�CxLͿ���(�@G�Aՙ�Cv�                                    Bx�:H�  T          @�33����?W
=A�HCx�\�����@
�HA�
=Cv��                                    Bx�:W,  �          @���������?=p�A��C{
������@ffA�  Cyh�                                    Bx�:e�  �          @�z��\���>�{@uCy� ��\��  ?�p�A�
=Cx}q                                    Bx�:tx  �          @��H�������R<�>�33Cx}q������  ?��A�
Cw�{                                    Bx�:�  �          @�������>��@9��Cz�
������?��A��Cy�                                     Bx�:��  �          @�33��\����>��?�33Cy����\����?\A���Cx��                                    Bx�:�j  �          @�(���\����?��@ָRCy�)��\��z�?�Q�A��\Cx
=                                    Bx�:�  �          @�(���=q��?�G�A5G�Cx����=q��{@�
A�(�Cvff                                    Bx�:��  �          @�33��z���G�?uA-��C}����z����@�\A�(�C{�                                    Bx�:�\  �          @�33������ff?�p�A�33C~E�������\@1G�B=qC|�                                    Bx�:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�:�   �          @������
=?��
A:�\Cz�=����\)@�A�G�Cx}q                                    Bx�:�N  �          @�(��У����H?�{A�z�CzuÿУ��|(�@6ffB�Cw�\                                    Bx�;�  �          @�33���H��Q�?�\)A��HCy:���H�w�@5Bp�Cv!H                                    Bx�;�  �          @���޸R��Q�?�{A���Cx�ÿ޸R�w�@4z�B�Cuٚ                                    Bx�;$@  �          @��
��33��(�?��HA���CuW
��33�Z=q@C33B�Cp�                                    Bx�;2�  �          @�zῴz����R?��HA�Q�C}c׿�z����@.�RB z�C{�                                    Bx�;A�  �          @�������ff?�(�AYG�C��{������@#�
A��C�=q                                    Bx�;P2  �          @�Q�J=q��=q?�AEG�C��ͿJ=q����@'�A�G�C�T{                                    Bx�;^�  �          @��׿����H?�  AQ��C��������@,(�A��
C���                                    Bx�;m~  �          @�  �Tz����H?xQ�A"�HC����Tz���33@�HA�\)C�+�                                    Bx�;|$  �          @�=q�c�
���?�{A8  C�E�c�
���H@#�
A�\)C��f                                    Bx�;��  �          @�G���\)���?5@�(�C}Y���\)��  @�A�Q�C|�                                    Bx�;�p  T          @������R���>��@���CyǮ���R��=q?���A��Cx�=                                    Bx�;�  �          @��
�����>��@��C{�R����
=?�
=A�=qCz��                                    Bx�;��  B          @�33��33���<��
>uC}z��33���H?ǮAy�C|�{                                    Bx�;�b  �          @��Ϳ�=q���?���AY��C�C׿�=q���R@7
=A�RC~�q                                    Bx�;�  T          @���������R?�33A��C�!H��������@E�B �C�XR                                    Bx�;�  
�          @�(��c�
���?�\)AX��C��׿c�
����@6ffA��C�                                      Bx�;�T  T          @�p��@  ����?p��AC�S3�@  ��=q@\)Aƣ�C�                                      Bx�;��  
�          @�z�p����Q�?=p�@�RC�aH�p�����H@�A�z�C�                                    Bx�<�  "          @��Ϳs33����?+�@ҏ\C�Y��s33���
@{A�33C��                                    Bx�<F  �          @�p�������>���@z=qC��ῇ����?�(�A�33C��                                    Bx�<+�  "          @���aG���=q?�@��C��\�aG����R@�A��RC�g�                                    Bx�<:�  T          @���(����H?�R@���C����(����R@
�HA��HC���                                    Bx�<I8  
Z          @�z�!G���=q>��@�=qC���!G���\)@G�A���C��q                                    Bx�<W�  "          @�{��{���?5@ۅC~)��{��Q�@p�A��RC|�R                                    Bx�<f�  �          @�p��������?B�\@�C�Ῑ����=q@�A�G�C���                                    Bx�<u*  �          @���u����>��H@���C�H��u���R@�A��RC���                                    Bx�<��  
�          @�z�^�R���>���@O\)C����^�R��Q�?�\)A�z�C�~�                                    Bx�<�v  "          @�33���
��  >�z�@4z�C�����
���R?�A�Q�C���                                    Bx�<�  
Z          @��\�z�H��\)>B�\?��C�,Ϳz�H��\)?��HA��RC��3                                    Bx�<��  T          @�녿���>aG�@{C�.�����?�(�A�Q�C��                                    Bx�<�h  T          @�=q�k����=�?��HC�n�k���Q�?��A���C�<)                                    Bx�<�  
�          @�녿\(����=u?!G�C����\(�����?�=qA|z�C��\                                    Bx�<۴  
�          @�Q쿦ff��33�����A�C�s3��ff��\)?�
=A=C�P�                                    Bx�<�Z  �          @�
=��Q���  �   ����CLͿ�Q���{?xQ�A�C#�                                    Bx�<�   �          @�{�����{�}p���
C�������Q�>�@�G�C�&f                                    Bx�=�  �          @��R���H��
=�333����C
=���H���R?B�\@���C                                    Bx�=L  T          @���{���^�R���C���{��
=?z�@��C��                                    Bx�=$�  
�          @�{��=q��  �z�����C�0���=q���R?^�RA��C�%                                    Bx�=3�  
Z          @�Q쿌������#�
���
C�t{�������\?W
=A
=C�l�                                    Bx�=B>  T          @�=q���
���R����Q�C�⏿��
��z�?��\A ��C��3                                    Bx�=P�  �          @�33�z�H��  �W
=�33C�0��z�H���?��AL(�C��                                    Bx�=_�  T          @���5��=q>�?�(�C��׿5���H?�\)A��RC�`                                     Bx�=n0  �          @���Tz����H>��?�z�C����Tz����?�33A��C��\                                    Bx�=|�  
�          @�p��\(���33>#�
?˅C�˅�\(����?�z�A��RC���                                    Bx�=�|  �          @�ff������=��
?E�C��ÿ����z�?�=qAv�RC��                                    Bx�=�"  �          @�ff�s33���>��@\)C�h��s33���?�33A��C�*=                                    Bx�=��  �          @�p��k����\>��
@I��C����k�����?�ffA�=qC�P�                                    Bx�=�n  �          @�z�!G����H>�z�@2�\C���!G����\?�\A��C���                                    Bx�=�  �          @�p�����(�>��@\)C��f�����\?��A��C��f                                    Bx�=Ժ  �          @��
=q��(�>��H@�{C�` �
=q��=q?��HA���C�9�                                    Bx�=�`  T          @�p��&ff���H?(��@���C��
�&ff���@�A��C���                                    Bx�=�  �          @�p�������\?@  @�\C�
=�����ff@��A�33C���                                    Bx�> �  
�          @�p������?
=@�\)C�w
������@33A�Q�C�O\                                    Bx�>R  
l          @����H���
?#�
@�{C������H����@ffA�C�}q                                    Bx�>�  t          @�
=�z�����?.{@���C�1�z���G�@��A�{C��                                    Bx�>,�  "          @�
=�����?333@���C�� ����G�@
=qA��C�W
                                    Bx�>;D  T          @�{��Q����?^�RA
=C�=q��Q����R@�
A�33C�)                                    Bx�>I�  �          @�ff��  ��(�?^�RA{C��;�  ��
=@�
A���C��
                                    Bx�>X�  
�          @��
��p���
=?�
=A9C�&f��p���  @$z�A�  C��)                                    Bx�>g6  4          @�33�
=q���?Y��Ap�C�T{�
=q��33@  A�\)C�"�                                    Bx�>u�  
�          @�ff��{����?8Q�@���C�T{��{��G�@
=qA��C�8R                                    Bx�>��  �          @�33��\��  ?c�
A  C�y���\��33@�\A�(�C�J=                                    Bx�>�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�>�t  
H          @��
>�  ����?@  A ��C�U�>�  ��@G�A�Q�C�l�                                    Bx�>�  �          @���?Ǯ���R���\�)C�W
?Ǯ����>�z�@AG�C�4{                                    Bx�>��  �          @���?�����H�n{�"�RC��f?����p�>�\)@C�
C�}q                                    Bx�>�f  "          @�=q?�Q�����������C�<)?�Q���p��\��=qC��)                                    Bx�>�  
�          @�z�?�G���
=��{��ffC�>�?�G����ÿ(�����C���                                    Bx�>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�?X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�?�  
�          @�\)?L��������\��C��H?L����녿��
�-C�G�                                    Bx�?%�  �          @��?�����������=qC�Q�?������\)��ffC��\                                    Bx�?4J  T          @���?�G����ÿ�p����C�XR?�G�����������\C��                                    Bx�?B�  �          @�G�?���{��(����HC��?����R����
=C���                                    Bx�?Q�  T          @�G�?�G�����z���C���?�G���Q�Tz��
{C�H                                    Bx�?`<  �          @��\?��H��녿�=q��G�C��?��H�����R�ϮC�z�                                    Bx�?n�  �          @���@{���H�   �ٮC���@{��녿���k�
C�e                                    Bx�?}�  T          @��@.�R����   ���HC��=@.�R���H��Q��r=qC�`                                     Bx�?�.  T          @�(�@\)���R�1G���=qC�N@\)��  ��Q����C��
                                    Bx�?��  �          @�p�@G����������\C�H@G���녿�{�5p�C�3                                    Bx�?�z  �          @���@-p���=q��z���  C���@-p����\�����{C��                                    Bx�?�   �          @���@J=q��33��G��r{C�B�@J=q���\�\�n�RC���                                    Bx�?��  T          @��@E���{��Q���{C�^�@E����׿W
=��C�h�                                    Bx�?�l  �          @��\@#�
�������H��Q�C��3@#�
��z�Tz��z�C�ٚ                                    Bx�?�  T          @��\@1���Q��p���=qC��\@1������33�?\)C�ff                                    Bx�?�  f          @�@����=q���ȣ�C���@����
=��z��P��C�g�                                    Bx�@^  �          @���@Fff�A������  C���@Fff�aG�����
=C���                                    Bx�@  �          @��R@Z=q�{�!G�����C�3@Z=q�?\)������
C�e                                    Bx�@�  �          @��@U�a녿�33���HC��R@U�p�׿
=q��(�C�Ф                                    Bx�@-P  �          @��R@I�����׿�G��V�\C�8R@I����ff�k��p�C���                                    Bx�@;�  �          @�G�@^�R���׿�(��:{C��@^�R��p���\)�+�C���                                    Bx�@J�  "          @�=q@=p���=q��33�7�
C��=@=p���ff    <#�
C�o\                                    Bx�@YB  �          @�z�@0�����H����� ��C���@0����{>L��?��C�xR                                    Bx�@g�  �          @��
@0  ���\��ff��\C���@0  ��>W
=@ ��C�s3                                    Bx�@v�  
�          @�@C�
��Q쿇��p�C�R@C�
���>B�\?޸RC��)                                    Bx�@�4  �          @�
=@1G���p����H�3�C���@1G����=�\)?+�C�E                                    Bx�@��  5          @�{@8Q����\���R�8��C�1�@8Q���
=<��
>L��C��                                    Bx�@��  
�          @�{@)����\)��z��,(�C��@)����33>�?�(�C���                                    Bx�@�&  �          @�\)@�\���R�z�H�(�C���@�\��G�>�Q�@S�
C���                                    Bx�@��  �          @�Q�?�{����k���RC��)?�{���R>�@�ffC���                                    Bx�@�r  �          @���?�p������ff��C�U�?�p���ff>��
@9��C�5�                                    Bx�@�  �          @���?�Q����Ϳz�H�33C��?�Q����R>Ǯ@dz�C�                                      Bx�@�  �          @���?�z���p��Tz���z�C��{?�z���ff?�@��
C��                                    Bx�@�d  �          @���@Q���33�L����G�C�f@Q���(�?��@�G�C���                                    Bx�A	
  �          @ə�?�z����R�=p���  C��?�z���
=?!G�@�
=C��q                                    Bx�A�  �          @�=q?�  ��녾u�	��C��?�  ���R?��A&�\C�33                                    Bx�A&V  �          @˅?�
=��  �B�\��(�C��?�
=����?(�@���C��H                                    Bx�A4�  �          @�?�
=���
�u�
{C���?�
=��>�G�@|(�C���                                    Bx�AC�  �          @�p�?������+���  C�Ff?�����p�?=p�@��HC�H�                                    Bx�ARH  T          @�z�@���=q��z��(  C��@���>.{?��C��                                    Bx�A`�  �          @��@Q���33������C�\@Q���{>�\)@{C���                                    Bx�Ao�  �          @�\)@G���\)���\�z�C�^�@G����>�33@E�C�@                                     Bx�A~:  �          @�Q�@����R���\�z�C�ٚ@���G�>�{@>{C���                                    Bx�A��  �          @�\)@\)��z῀  ��\C�xR@\)���R>�{@>{C�W
                                    Bx�A��  �          @�(�@���G���{� ��C�c�@���z�>L��?�C�5�                                    Bx�A�,  �          @˅@��{�\(���  C��
@��\)>�@��C��f                                    Bx�A��  �          @��H?z�H�ƸR�z���\)C�~�?z�H��{?Q�@�p�C���                                    Bx�A�x  �          @�=q?W
=��  �.{�ǮC��?W
=��z�?�(�A1C��H                                    Bx�A�  �          @�G�?Q���
=��\)�$z�C��)?Q���(�?���A!G�C��=                                    Bx�A��  �          @ə�?8Q��Ǯ��Q��S�
C�G�?8Q���p�?��\A��C�Q�                                    Bx�A�j  �          @��?^�R�Ǯ������C��R?^�R���
?�  A6ffC��                                    Bx�B  �          @���?s33��ff���Ϳ^�RC�W
?s33��=q?��\A:�RC�o\                                    Bx�B�  �          @���?O\)��\)=�Q�?Q�C��
?O\)����?���AV{C���                                    Bx�B\  �          @ə�?Q��Ǯ>\)?���C���?Q����?�G�A]C��
                                    Bx�B.  �          @���>�=q��(��#�
��Q�C�8R>�=q��\)?�{AD(�C�>�                                    Bx�B<�  �          @�{<��
��p�>��@�C�
<��
��ff?�z�Ao�C�R                                    Bx�BKN  �          @����
����?!G�@��C��ͼ��
�Å@ ��A���C��                                    Bx�BY�  �          @�  >�����
=?
=q@�ffC�\)>�����ff?��A�  C�l�                                    Bx�Bh�  �          @ƸR>���z�?:�H@�Q�C�>�>����H@33A�(�C�\)                                    Bx�Bw@  �          @�?E����H?+�@�{C��H?E�����?�(�A��HC��\                                    Bx�B��  �          @ƸR?5��z�>�G�@�=qC�Q�?5����?�G�A��
C�s3                                    Bx�B��  �          @�ff?!G���(�?\)@�
=C���?!G����
?��A��C�
=                                    Bx�B�2  �          @�ff?\)���
?&ff@��C�� ?\)���H?���A��C��H                                    Bx�B��  �          @�p���Q�����>�  @33C��{��Q���ff?���Ak�
C���                                    Bx�B�~  �          @��>����
>��@��C�(�>����
?��
A�ffC�@                                     Bx�B�$  �          @��>�(���(�>���@1�C�  >�(���p�?�{Ar=qC��                                    Bx�B��  �          @�\)?������H?���A$��C���?�����
=@�A�C�                                    Bx�B�p  �          @Ϯ@ ����  ?�
=Ao\)C��\@ ����  @7�AѮC�Ǯ                                    Bx�B�  �          @�ff@ff��  ?�Q�As�C��@ff��Q�@8Q�Aԏ\C��                                    Bx�C	�  T          @θR@	����G�?��A�G�C�.@	����Q�@Dz�A�33C�q                                    Bx�Cb  �          @�(�@ff��z�?�=qA�33C�L�@ff���
@?\)A�p�C�P�                                    Bx�C'  �          @�ff@Q����\?�Az�\C�s3@Q����H@4z�A؏\C�XR                                    Bx�C5�  �          @�?�������?�=qAm�C��=?�����@/\)A��HC�S3                                    Bx�CDT  �          @�G�?�����H?���AO�
C��H?����p�@ ��A���C���                                    Bx�CR�  �          @�
=?�  ���?���A.�HC�g�?�  ��G�@33A��
C��                                    Bx�Ca�  �          @�  ?Y����(�?޸RA��C�E?Y����z�@8��A�ffC���                                    Bx�CpF  �          @�\)?�
=����?�A[�C�k�?�
=��(�@#�
A��HC�3                                    Bx�C~�  �          @��?޸R���
?z�HA33C��q?޸R��G�@��A�ffC��                                    Bx�C��  �          @�{?��R���?���AW
=C��?��R��  @\)A�ffC��)                                    Bx�C�8  �          @�
=?�(���z�?�\)A{�
C��?�(���@-p�AظRC��H                                    Bx�C��  �          @��R?�Q�����?�  A��C��?�Q���G�@7�A�(�C���                                    Bx�C��  �          @�p�?��H��\)@ffA�G�C��?��H��p�@I��BffC��                                    Bx�C�*  �          @�
=?k���ff?�  AiC���?k���  @*=qA�=qC���                                    Bx�C��  �          @���?z�H��  ?O\)@��
C�ٚ?z�H��ff?��RA�{C�)                                    Bx�C�v  �          @�z�>������?�  Al��C�  >�����
=@)��A�z�C�+�                                    Bx�C�  �          @��=#�
��Q�?�{AK�C�1�=#�
��33@%�A�{C�5�                                    Bx�D�  �          @��
���
��
=?���A?\)C��׼��
���@%�A��
C��H                                    Bx�Dh  �          @�(��\����?z�HA�C�Ff�\��{@G�A���C�,�                                    Bx�D   �          @�=q>���Q������C���>���(�?��\A;
=C��q                                    Bx�D.�  �          @���>���  ������C��)>���z�?�
=A-p�C���                                    Bx�D=Z  �          @�G�?G���ff����C���?G����?W
=@�C��                                     Bx�DL   �          @��>�(���33>�@��\C���>�(��Å?��
A��C�                                    Bx�DZ�  �          @�
=>L����(�?��
A�\C�� >L������@z�A���C���                                    Bx�DiL  T          @�
=>k���p�?8Q�@˅C��>k���(�@G�A�{C��                                    Bx�Dw�  �          @Ϯ>�
=��z�?W
=@�RC�� >�
=�\@��A��\C��R                                    Bx�D��  �          @�\)?h����{?��Aj�RC�'�?h����
=@8Q�A�\)C�}q                                    Bx�D�>  �          @�p�?xQ���
=@Q�A�
=C���?xQ����@S�
A��\C�{                                    Bx�D��  �          @��?Tz��\?���A�(�C��?Tz���=q@A�A�  C�=q                                    Bx�D��  �          @���>���33?�=qA��HC�*=>���33@B�\A�G�C�\)                                    Bx�D�0  �          @�(�=�\)����@z�A�C�W
=�\)��\)@P��A�Q�C�`                                     Bx�D��  �          @��;�����Q�@
�HA���C��
������{@VffA���C�p�                                    Bx�D�|  �          @�=q�Y�����@  A��C��R�Y����\)@X��Bp�C�c�                                    Bx�D�"  �          @θR��33���@(�A��RC�Y���33���H@VffA�(�CT{                                    Bx�D��  �          @�ff��z���\)?�(�Aw33C~� ��z���Q�@9��A�{C}:�                                    Bx�E
n  T          @�{�33��p�@   A�z�Cw�f�33����@G
=A�
=Cu�f                                    Bx�E  �          @��{���H?�G�AYCy0��{���@*�HA�=qCw��                                    Bx�E'�  �          @�������
=?�(�A1C}LͿ����33@=qA��\C|@                                     Bx�E6`  �          @�\)�333��(�?333@ϮC����333���?�
=A���C��R                                    Bx�EE  �          @�\)�}p���33?(��@\C�aH�}p����H?��A���C�.                                    Bx�ES�  �          @ȣ׿��\��p�>�ff@���C�LͿ��\��ff?�Q�Az�HC�!H                                    Bx�EbR  �          @ə���
=���
>�G�@\)C�lͿ�
=����?�Au��C�1�                                    Bx�Ep�  �          @��þ\��(�?��
A(�C�<)�\��G�@  A�33C�"�                                    Bx�E�  �          @���>k���(�?��RA7�C��>k���  @p�A���C�"�                                    Bx�E�D  �          @ȣ�>.{��33?�{A%G�C���>.{��  @z�A��C�ٚ                                    Bx�E��  �          @�=q�&ff��ff?s33A
�HC���&ff��(�@�A��C�ٚ                                    Bx�E��  �          @ə�?�
=��=q@��A�p�C�9�?�
=��
=@^{B�\C�                                    Bx�E�6  �          @���@	����G�@\(�B�
C��@	����  @��\B.�C�&f                                    Bx�E��  �          @�G�?����@4z�A֣�C���?���(�@xQ�Bz�C��                                    Bx�Eׂ  T          @�(�?�=q��{@^{B	�HC�&f?�=q��z�@�z�B6��C�J=                                    Bx�E�(  �          @Ǯ<#�
��p�?��RA��\C��<#�
����@E�A��C��                                    Bx�E��  �          @ə��u��z�?�G�A�C����u���@{A��C�L�                                    Bx�Ft  �          @ȣ׿J=q��33?��A z�C�S3�J=q��  @�\A��
C�)                                    Bx�F  �          @ə��#�
��?fffA\)C�f�#�
��(�@�A��C��                                     Bx�F �  �          @ʏ\��Q����@p�A��C�C׾�Q���33@UA�C��                                    Bx�F/f  �          @�  ��\)���
@A���C��q��\)���\@N{A���C�z�                                    Bx�F>  �          @�(��B�\��\)?��A z�C��B�\����@�RA�=qC��                                    Bx�FL�  �          @�\)�+�����@�RA�Q�Cp�\�+���Q�@H��A���Cm�\                                    Bx�F[X  �          @�{�G���Q�@
�HA�=qCu���G���\)@HQ�A�33Cs�                                    Bx�Fi�  �          @����7����\@�A�\)CoL��7����@B�\A�33Cl^�                                    Bx�Fx�  �          @�Q��G����
@
=qA�ffCv{�G����H@HQ�A���Cs��                                    Bx�F�J  �          @��R��\��{@�HA�(�Cu&f��\���@W
=B	=qCrO\                                    Bx�F��  �          @�������@��A�Q�CT{�������@I��B\)C}Ǯ                                    Bx�F��  �          @�{�z�H���R@�
A��RC����z�H��{@FffA��RC�c�                                    Bx�F�<  �          @�Q�����ff?�33A���C�B�������R@?\)A�
=C�0�                                    Bx�F��  T          @����Q���\)@�A��C�K���Q���@Tz�A�33C��                                    Bx�FЈ  �          @��H=�\)����?���A�{C�Q�=�\)���@@��A�p�C�XR                                    Bx�F�.  �          @�Q�?�p���G�@z�A��\C���?�p���Q�@K�A�C��                                    Bx�F��  �          @��@ff����@{AîC��3@ff��=q@X��B
z�C�N                                    Bx�F�z  �          @�
=@   ��p�?�=qAO\)C�"�@   ����@Q�A��HC��\                                    Bx�G   T          @Å?�Q���
=?�Q�A4��C�=q?�Q����
@33A��C��q                                    Bx�G�  �          @���?�
=��
=?���A�33C���?�
=��  @8��A�  C�L�                                    Bx�G(l  �          @\@G���z�?��A�33C�G�@G���p�@:�HA�  C�/\                                    Bx�G7  �          @�(�@%���\)?��Ap�C�q@%�����@��A�G�C��H                                    Bx�GE�  �          @�z�?�z���
=?�p�A_�
C�
?�z���=q@%�A���C���                                    Bx�GT^  �          @�=q@  ���?�A�Q�C��@  ��z�@<��AָRC�\)                                    Bx�Gc  �          @�G�@����z�@!�A��C�b�@������@eB�C���                                    Bx�Gq�  �          @��ÿ:�H����@0��AҸRC�=q�:�H���
@r�\BQ�C��                                     Bx�G�P  �          @��Ϳ�  ��33@333A�ffC��R��  ��{@vffB\)C��                                    Bx�G��  �          @ȣ׿�=q��z�@1G�A��
C}����=q��  @q�B=qC{�\                                    Bx�G��  �          @�33�h�����\@z�A���C�` �h����Q�@W�B  C��q                                    Bx�G�B  �          @��
�Ǯ���R@��A��HC~�Ǯ���
@^�RB  C}�                                    Bx�G��  �          @�{������
@�
A�Q�C��\�������@Z�HB G�C�+�                                    Bx�GɎ  �          @\�Ǯ���H@Q�A�G�C}���Ǯ����@XQ�B�HC{�)                                    Bx�G�4  �          @��׿��\��@5A�ffC�j=���\����@s33Bp�C��                                     Bx�G��  �          @��R>B�\��G�@,(�A�G�C��>B�\���@j�HB�
C�0�                                    Bx�G��  �          @�33?��
��
=@�A��
C���?��
���@VffB33C�=q                                    Bx�H&  T          @��\?�����@@��A��C�3?���{�@vffB%��C�                                    Bx�H�  �          @��?�ff���H@6ffA�{C��q?�ff��{@o\)B��C�#�                                    Bx�H!r  �          @���@
�H��@`  B�C��H@
�H�Z=q@�  B9�C�B�                                    Bx�H0  �          @��@E��.�R@�=qB2��C�:�@E����@���BM�C�P�                                    Bx�H>�  �          @��
?�������@J�HA��C�Z�?�����{@��B&\)C��\                                    Bx�HMd  �          @�(����
���H?��HA���C|Y����
����@1G�A�G�Cz�                                    Bx�H\
  �          @��H��R���?\A]Cx����R���\@'
=A\CwE                                    Bx�Hj�  �          @��
�
=����?W
=@�ffCv�)�
=��  ?�33A�=qCu��                                    Bx�HyV  �          @ȣ׿���z�?�ffA@z�C~33������@�A���C}5�                                    Bx�H��  �          @�\)����?�R@�{Cv޸����ff?��HA�(�Cv�                                    Bx�H��  �          @�z��  ���>�?��RC}E��  ��p�?�G�A@Q�C|��                                    Bx�H�H  �          @�z��������?   @���Cx+������ff?˅AqG�Cwn                                    Bx�H��  
�          @��������?�z�A2�HC��Ϳ�����@��A�G�C�K�                                    Bx�H  �          @�z�?\)��
=?�p�A�(�C���?\)��\)@C�
A�C��                                    Bx�H�:  �          @��H=u���H?ٙ�A�(�C�O\=u��z�@3�
A�(�C�U�                                    Bx�H��  �          @�33=#�
���?��RA�
=C�/\=#�
���@Dz�A��C�4{                                    Bx�H�  �          @�녿\)����?���Ar{C�8R�\)��(�@+�Aҏ\C�H                                    Bx�H�,  �          @����s33��=q?��AI�C�XR�s33��ff@�HA�\)C�
=                                    Bx�I�  �          @���\(�����?���AK�C����\(�����@�HA���C�u�                                    Bx�Ix  �          @�\)>�z�����@(�A�G�C��H>�z���\)@N�RB�\C��=                                    Bx�I)  �          @�
=>8Q���ff@��A�  C��
>8Q���(�@Z�HB��C�3                                    Bx�I7�  �          @���>�����@�A�  C�t{>����@Tz�B�
C��R                                    Bx�IFj  �          @�G�>���p�@   A��C�k�>���p�@Dz�A���C���                                    Bx�IU  �          @��?\)���@��A�C��H?\)����@S33B��C�0�                                    Bx�Ic�  �          @���?=p����
@+�A�=qC��?=p����@j�HB�C�h�                                    Bx�Ir\  �          @���?Tz���z�@?\)A�G�C���?Tz����R@|(�B$  C�K�                                    Bx�I�  �          @�z�?����=q@HQ�B(�C�j=?���w�@~{B2�
C��f                                    Bx�I��  �          @��
?�z���z�@g
=B��C��\?�z��W
=@�33BJ��C���                                    Bx�I�N  �          @��?^�R��p�@HQ�B (�C��?^�R��
=@���B,\)C��{                                    Bx�I��  �          @�z�?B�\���@8��A�\)C�K�?B�\����@tz�B"�C���                                    Bx�I��  �          @��
�(�����R?���A)G�C���(����z�@
�HA�{C�|)                                    Bx�I�@  �          @��
�333���
?�ffA((�C�t{�333���@Q�A�\)C�@                                     Bx�I��  �          @�
=>�  ��?�  A��RC�AH>�  ��\)@4z�A��C�]q                                    Bx�I�  �          @�
=>��
��Q�?�(�AeG�C��3>��
���@$z�Ȁ\C���                                    Bx�I�2  �          @���=�Q����?�\)AR�\C�w
=�Q���\)@\)A�G�C�~�                                    Bx�J�  �          @\��Q����
?��HA^�\C�:ᾸQ���\)@%�A�33C��                                    Bx�J~  �          @��
���
���H?�\A���C�p����
��(�@8Q�A��HC�O\                                    Bx�J"$  �          @Å>����=q?���A�z�C��
>�����@:�HA�\C��f                                    Bx�J0�  T          @���?@  ���@"�\AǅC��3?@  ���@c33B\)C�o\                                    Bx�J?p  �          @���?��\��=q@HQ�A�\)C���?��\���@��B(C��
                                    Bx�JN  �          @�\)?u��
=@J�HB G�C�s3?u��  @��\B,p�C�W
                                    Bx�J\�  �          @�  ?p������@Tz�Bp�C�u�?p������@��RB2��C�g�                                    Bx�Jkb  �          @��R?�33���@Y��B�\C���?�33�~�R@�Q�B7(�C�                                    Bx�Jz  �          @�(�?������@N�RBz�C��)?����u@��B.�HC���                                    Bx�J��  �          @�?����{@^{BQ�C�Ff?���z�H@�=qB;33C�t{                                    Bx�J�T  �          @�\)?5��  @H��A�
=C��?5��G�@��B,(�C���                                    Bx�J��  �          @�z�?��\��33@-p�A�(�C��f?��\��
=@j=qBG�C�k�                                    Bx�J��  �          @�\)?h�����
@EB�C���?h���z�H@{�B1Q�C��                                     Bx�J�F  �          @��R�u����@   A�
=C����u��ff@\(�B��C�p�                                    Bx�J��  �          @���@L���HQ�@Mp�Bp�C�ٚ@L���(�@p��B,{C�^�                                    Bx�J��  T          @��
?�Q����@@  B�HC�AH?�Q��dz�@qG�B+p�C�8R                                    Bx�J�8  �          @�=q@��33@G
=B=qC���@�Z=q@vffB/C�Ǯ                                    Bx�J��  
�          @��
@�����@3�
A�C��
@��j�H@fffB(�C��                                    Bx�K�  
�          @��H@����p�@5�A�(�C��@���a�@fffB ��C�f                                    Bx�K*  T          @���@�����@8Q�A�(�C�` @���^{@hQ�B%
=C��f                                    Bx�K)�  �          @��H@
=q��33@�A�G�C��
@
=q���@H��B	=qC��                                    Bx�K8v  �          @�Q�@��xQ�@?\)BG�C�
=@��N{@l(�B)�RC��=                                    Bx�KG  �          @�=q@h�ÿ��@FffB�C�P�@h�ÿ��R@Z=qB&ffC��q                                    Bx�KU�  �          @���@�=q?}p�@�A�{AD(�@�=q?�
=@A��RA�G�                                    Bx�Kdh  �          @��H@�\)@8��?�  A��HB	\)@�\)@HQ�?h��A(�B                                      Bx�Ks  �          @�z�@���@B�\?��A�  B
=@���@Vff?�  AV�RB��                                    Bx�K��  �          @�z�@q�@R�\?�Q�A���B$(�@q�@g
=?�  AW
=B.�                                    Bx�K�Z  T          @��@��H@�
@��Aأ�A��
@��H@/\)?�33A��RBQ�                                    Bx�K�   T          @�@�G�?�=q@%�A�p�A�p�@�G�@�
@
�HA��
A�                                      Bx�K��  	A          @��R@��?�=q@,��A��A��
@��@�@�\A�ffA�ff                                    Bx�K�L  
y          @�\)@��H?�  @/\)A�
=A���@��H@  @A���A��                                    Bx�K��  
�          @��
@q�?�(�@FffB�\A�(�@q�@33@/\)BG�A�Q�                                    Bx�K٘  �          @��@l(�?�p�@Dz�BA��H@l(�@33@*=qA�A��                                    Bx�K�>  �          @�(�@z=q?�ff@AG�B�A��@z=q?��@-p�A��
A�33                                    Bx�K��  �          @��@�?(�@5B�HAQ�@�?�z�@*�HA���Ax(�                                    Bx�L�  �          @�33@�p�>\)@:=qB�?���@�p�?8Q�@5�B��A                                      Bx�L0  �          @�=q@u���R@R�\B"{C���@u>�33@R�\B!�
@�ff                                    Bx�L"�  �          @��\@}p���@G�B
=C��@}p�>��@I��B  @ff                                    Bx�L1|  �          @��R@\)�\@8��B33C�@ @\)>L��@:=qB=q@6ff                                    Bx�L@"  �          @���@��
��
=@%B 
=C�q@��
=�G�@'�B?��R                                    Bx�LN�  
�          @�
=@[�����@Z=qB-z�C��f@[���(�@c�
B7��C�u�                                    Bx�L]n  �          @��H@E��
@EB��C�/\@E��33@\(�B5\)C�Ф                                    Bx�Ll  �          @��H@a녿�@1G�B�HC�g�@a녿�(�@E�B{C�q�                                    Bx�Lz�  �          @��@�����
=@�RA�
=C���@����5@�HA�{C���                                    Bx�L�`  �          @��
@u�!G�@�A��C�T{@u��@   BffC�H                                    Bx�L�  �          @�  @b�\�\)@EB#{C�|)@b�\=�\)@H��B&G�?�=q                                    Bx�L��  �          @�=q@Z�H��\)@UB0��C���@Z�H>���@U�B0(�@�33                                    Bx�L�R  �          @�(�@@  ��\@G
=B'  C��\@@  ��\)@Y��B;
=C���                                    Bx�L��  �          @�ff?�\)����?
=@��C�E?�\)���?˅A���C���                                    Bx�LҞ  �          @�33?&ff����?��@�
=C���?&ff����?�33A���C���                                    Bx�L�D  �          @�?������?�  A@z�C��f?���\)?�=qA��\C��f                                    Bx�L��  �          @���@K��!G�?��A���C��
@K��(�?�{A�  C�                                    Bx�L��  �          @��R@QG���@�A��
C�˅@QG����H@�HBG�C��{                                    Bx�M6  T          @��@L���33@(�B  C��=@L�Ϳ\@3�
B��C�O\                                    Bx�M�  �          @�33@0  �!�@*=qB{C���@0  ��Q�@G�B+ffC�j=                                    Bx�M*�  �          @��
@333���H@<(�B+��C�AH@333�W
=@K�B=ffC��3                                    Bx�M9(  �          @�
=@B�\���@C33B-�
C�j=@B�\��
=@L��B8��C�3                                    Bx�MG�  �          @���@\�Ϳ�
=@'
=B��C��@\�Ϳ!G�@333Bz�C��\                                    Bx�MVt  �          @���@a녾�(�@1G�BC��@a�>\)@333B��@z�                                    Bx�Me  �          @�z�@g��c�
@.{B�C�{@g����
@5B=qC�t{                                    